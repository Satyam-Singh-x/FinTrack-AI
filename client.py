from __future__ import annotations

import asyncio
import os
import sys
import textwrap
import traceback
import uuid
from datetime import date
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

SERVER_PATH  = Path(os.getenv("MCP_SERVER_PATH", str(Path(__file__).parent / "server.py")))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
GROQ_MODEL   = os.getenv("GROQ_MODEL",   "llama-3.3-70b-versatile")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")        # ← any model you have pulled
OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://localhost:11434")
THREAD_ID    = str(uuid.uuid4())
TODAY        = date.today().isoformat()

_NO_COLOUR = not sys.stdout.isatty() or os.getenv("NO_COLOR")
def _c(code, text): return text if _NO_COLOUR else f"\033[{code}m{text}\033[0m"
def cyan(t):   return _c("36", t)
def green(t):  return _c("32", t)
def yellow(t): return _c("33", t)
def dim(t):    return _c("2",  t)
def bold(t):   return _c("1",  t)
def red(t):    return _c("31", t)
def magenta(t):return _c("35", t)

SYSTEM_PROMPT = f"""You are an expert personal finance assistant with full access to a MySQL-backed expense tracker through MCP tools. Today is {TODAY}.

Help users log expenses/income/savings, set budgets, analyse trends, and answer finance questions.

TOOLS: add_expense, add_expenses_bulk, get_expense_by_id, list_expenses, search_expenses, update_expense, delete_expense_by_id, delete_expenses_by_filter, get_categories, summarize_by_category, summarize_by_period, get_top_expenses, get_cashflow, get_monthly_trend, set_budget, set_budgets_bulk, list_budgets, get_budget_by_id, update_budget, delete_budget, get_budget_status, get_budget_vs_actual, get_overbudget_categories, get_budget_trend, suggest_budgets_from_history, copy_budgets_from_month, add_saving, set_savings_goal, get_savings_summary, get_savings_trend

RULES:

1. Always call get_categories before adding expenses.

2. Infer: Food / Transport / Entertainment / Health / Utilities.

3. Use add_expenses_bulk for lists.

4. Currency: rupees. today={TODAY}, this month={TODAY[:7]}.

5. Confirm writes with ID + category.

6. Format amounts as rupees 1,200.50.

7. Never hallucinate data.

IMPORTANT TOOL CALL FORMAT:

When calling a tool, ONLY return JSON like:

{{
  "month": "2025-03"
}}

DO NOT return:
- "function"
- "args"
- nested structures

TOOL CALLING FORMAT (when using Ollama locally):
- Always use proper JSON tool call format.
- Never skip required fields in tool arguments.
- Call one tool at a time and wait for the result before proceeding."""


QUOTA_MARKERS = [
    "quota", "rate limit", "resource exhausted", "429",
    "too many requests", "quota exceeded", "rate_limit_exceeded",
    "insufficient_quota", "billing", "exceeded your",
"organization has been restricted",
        "invalid_request_error",
]

def _is_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(m in msg for m in QUOTA_MARKERS)


# ── Fallback chain: Gemini ──► Groq ──► Ollama ─────────────────────────────

class FallbackLLM:
    """
    Wraps up to three LLMs in a priority chain.
    On a quota / rate-limit error the chain advances to the next model.
    All three slots support bind_tools so tool-calling works at every tier.
    """

    TIER_LABELS = ["Gemini", "Groq", "Ollama"]

    def __init__(self, models: list):
        """
        models: ordered list [primary, secondary, tertiary]
                (tertiary / Ollama is optional but recommended)
        """
        if not models:
            raise ValueError("At least one model required.")
        self._models = models
        self._tier   = 0          # index into self._models

    # ── current active model ───────────────────────────────────────────────

    @property
    def _active(self):
        return self._models[self._tier]

    def _label(self, tier: int | None = None) -> str:
        idx = self._tier if tier is None else tier
        labels = self.TIER_LABELS
        return labels[idx] if idx < len(labels) else f"model-{idx}"

    # ── advance to next tier ───────────────────────────────────────────────

    def _advance(self, exc: Exception) -> bool:
        """Try to move to the next tier. Returns True if advanced, False if exhausted."""
        if self._tier < len(self._models) - 1:
            self._tier += 1
            print(dim(f"\n  [fallback] {self._label(self._tier - 1)} quota hit "
                      f"— switching to {magenta(self._label())}.\n"))
            return True
        print(red(f"\n  [fallback] All models exhausted. Last error: {exc}\n"))
        return False

    # ── bind_tools must propagate through the entire chain ─────────────────

    def bind_tools(self, tools, **kwargs):
        bound_models = []
        for m in self._models:
            try:
                bound_models.append(m.bind_tools(tools, **kwargs))
            except Exception:
                # Ollama may not support every kwarg; try bare bind_tools
                try:
                    bound_models.append(m.bind_tools(tools))
                except Exception:
                    bound_models.append(m)          # last resort: unbounded
        new = FallbackLLM(bound_models)
        new._tier = self._tier
        return new

    # ── async invoke ───────────────────────────────────────────────────────

    async def ainvoke(self, messages, config=None, **kwargs):
        while True:
            try:
                return await self._active.ainvoke(messages, config=config, **kwargs)
            except Exception as exc:
                if _is_quota_error(exc) and self._advance(exc):
                    continue
                raise

    # ── sync invoke ────────────────────────────────────────────────────────

    def invoke(self, messages, config=None, **kwargs):
        while True:
            try:
                return self._active.invoke(messages, config=config, **kwargs)
            except Exception as exc:
                if _is_quota_error(exc) and self._advance(exc):
                    continue
                raise


# ── LLM factory ────────────────────────────────────────────────────────────

def _build_llm() -> FallbackLLM:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_groq        import ChatGroq
    from langchain_ollama      import ChatOllama          # pip install langchain-ollama

    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key   = os.getenv("GROQ_API_KEY")

    if not gemini_key:
        raise RuntimeError("GEMINI_API_KEY not set in .env")
    if not groq_key:
        raise RuntimeError("GROQ_API_KEY not set in .env")

    gemini = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0,
        google_api_key=gemini_key,
    )

    groq = ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        groq_api_key=groq_key,
    )

    # Ollama — runs locally, no key needed.
    # tool_call_format="json" keeps tool-use reliable on most Ollama models.
    ollama = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_URL,
        temperature=0,
        # Ensures structured tool-call output even on models without native
        # function-calling support (falls back to JSON extraction).
        format="json",
    )

    print(dim(f"  LLM chain  →  Gemini/{GEMINI_MODEL}  →  "
              f"Groq/{GROQ_MODEL}  →  Ollama/{OLLAMA_MODEL}"))

    return FallbackLLM(models=[gemini, groq, ollama])


# ── LangGraph state & graph ─────────────────────────────────────────────────

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


async def build_graph(tools):
    llm            = _build_llm()
    llm_with_tools = llm.bind_tools(tools)

    async def chat_node(state):
        msgs = state["messages"]
        if not msgs or not isinstance(msgs[0], SystemMessage):
            msgs = [SystemMessage(content=SYSTEM_PROMPT)] + list(msgs)
        response = await llm_with_tools.ainvoke(msgs)
        return {"messages": [response]}

    tool_node = ToolNode(tools)
    graph     = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tool_node", tool_node)
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges(
        "chat_node", tools_condition,
        {"tools": "tool_node", END: END},
    )
    graph.add_edge("tool_node", "chat_node")
    return graph.compile(checkpointer=InMemorySaver())


# ── helpers ─────────────────────────────────────────────────────────────────

def _extract_text(content) -> str:
    if isinstance(content, str):   return content.strip()
    if isinstance(content, list):
        parts = [b.get("text", "") if isinstance(b, dict) else str(b) for b in content]
        return "\n".join(p for p in parts if p).strip()
    return str(content).strip()


# ── REPL ─────────────────────────────────────────────────────────────────────

async def run_repl(graph, tools):
    thread_id  = THREAD_ID
    thread_cfg = {"configurable": {"thread_id": thread_id}}

    print(bold(cyan("\n  Expense Tracker  Finance AI")))
    print(dim(f"  /help for commands\n"))

    while True:
        try:
            raw = input(bold(yellow("you  >  "))).strip()
        except (EOFError, KeyboardInterrupt):
            print(dim("\n  Bye!\n")); break

        if not raw: continue
        if raw.lower() in {"exit", "quit", "/exit", "/quit"}:
            print(dim("  Bye!\n")); break
        if raw == "/help":
            print("  /tools  /history  /clear  /exit"); continue
        if raw == "/tools":
            for t in tools:
                print(f"  {cyan(getattr(t, 'name', str(t)))}")
            continue
        if raw == "/clear":
            thread_id  = str(uuid.uuid4())
            thread_cfg = {"configurable": {"thread_id": thread_id}}
            print(dim("  Cleared.\n")); continue

        print(bold(green("ai   >  ")), end="", flush=True)
        try:
            result  = await graph.ainvoke(
                {"messages": [HumanMessage(content=raw)]},
                config=thread_cfg,
            )
            content = getattr(result["messages"][-1], "content", "")
            print(textwrap.fill(
                _extract_text(content), width=90,
                subsequent_indent="        ",
                break_long_words=False, break_on_hyphens=False,
            ))
        except Exception as exc:
            print(red(f"\n  [error] {exc}"))
            if os.getenv("DEBUG"):
                traceback.print_exc()
        print()


# ── entry point ───────────────────────────────────────────────────────────────

async def main():
    if not SERVER_PATH.exists():
        print(red(f"[fatal] server.py not found: {SERVER_PATH}"))
        sys.exit(1)

    print(dim(f"  Connecting to {SERVER_PATH} ..."))
    mcp_client = MultiServerMCPClient({
        "expense_tracker": {
            "transport": "stdio",
            "command":   sys.executable,
            "args":      [str(SERVER_PATH)],
        }
    })
    tools = await mcp_client.get_tools()
    print(dim(f"  {len(tools)} tools loaded.\n"))
    graph = await build_graph(tools)
    await run_repl(graph, tools)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
