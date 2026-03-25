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
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GROQ_MODEL   = os.getenv("GROQ_MODEL",   "llama-3.3-70b-versatile")
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
7. Never hallucinate data."""

QUOTA_MARKERS = ["quota", "rate limit", "resource exhausted", "429", "too many requests", "quota exceeded"]

def _is_quota_error(exc):
    return any(m in str(exc).lower() for m in QUOTA_MARKERS)


class FallbackLLM:
    """Tries Gemini first, falls back to Groq on quota errors."""

    def __init__(self, primary, fallback):
        self._primary  = primary
        self._fallback = fallback
        self._using_fallback = False

    def bind_tools(self, tools, **kwargs):
        bound = FallbackLLM(
            primary=self._primary.bind_tools(tools, **kwargs),
            fallback=self._fallback.bind_tools(tools, **kwargs),
        )
        bound._using_fallback = self._using_fallback
        return bound

    async def ainvoke(self, messages, config=None, **kwargs):
        if self._using_fallback:
            return await self._fallback.ainvoke(messages, config=config, **kwargs)
        try:
            return await self._primary.ainvoke(messages, config=config, **kwargs)
        except Exception as exc:
            if _is_quota_error(exc):
                print(dim("\n  [fallback] Gemini quota hit — switching to Groq."))
                self._using_fallback = True
                return await self._fallback.ainvoke(messages, config=config, **kwargs)
            raise

    def invoke(self, messages, config=None, **kwargs):
        if self._using_fallback:
            return self._fallback.invoke(messages, config=config, **kwargs)
        try:
            return self._primary.invoke(messages, config=config, **kwargs)
        except Exception as exc:
            if _is_quota_error(exc):
                self._using_fallback = True
                return self._fallback.invoke(messages, config=config, **kwargs)
            raise


def _build_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_groq import ChatGroq

    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key   = os.getenv("GROQ_API_KEY")
    if not gemini_key: raise RuntimeError("GEMINI_API_KEY not set in .env")
    if not groq_key:   raise RuntimeError("GROQ_API_KEY not set in .env")

    return FallbackLLM(
        primary=ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0, google_api_key=gemini_key),
        fallback=ChatGroq(model=GROQ_MODEL, temperature=0, groq_api_key=groq_key),
    )


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
    graph.add_conditional_edges("chat_node", tools_condition, {"tools": "tool_node", END: END})
    graph.add_edge("tool_node", "chat_node")
    return graph.compile(checkpointer=InMemorySaver())


def _extract_text(content):
    if isinstance(content, str):   return content.strip()
    if isinstance(content, list):
        parts = [b.get("text","") if isinstance(b,dict) else str(b) for b in content]
        return "\n".join(p for p in parts if p).strip()
    return str(content).strip()


async def run_repl(graph, tools):
    thread_id  = THREAD_ID
    thread_cfg = {"configurable": {"thread_id": thread_id}}

    print(bold(cyan("\n  Expense Tracker  Finance AI")))
    print(dim(f"  Primary: Gemini/{GEMINI_MODEL}  |  Fallback: Groq/{GROQ_MODEL}"))
    print(dim(f"  /help for commands\n"))

    while True:
        try:
            raw = input(bold(yellow("you  >  "))).strip()
        except (EOFError, KeyboardInterrupt):
            print(dim("\n  Bye!\n")); break

        if not raw: continue
        if raw.lower() in {"exit","quit","/exit","/quit"}:
            print(dim("  Bye!\n")); break
        if raw == "/help":
            print(f"  /tools  /history  /clear  /exit"); continue
        if raw == "/tools":
            for t in tools: print(f"  {cyan(getattr(t,'name',str(t)))}")
            continue
        if raw == "/clear":
            thread_id = str(uuid.uuid4())
            thread_cfg = {"configurable": {"thread_id": thread_id}}
            print(dim("  Cleared.\n")); continue

        print(bold(green("ai   >  ")), end="", flush=True)
        try:
            result  = await graph.ainvoke({"messages": [HumanMessage(content=raw)]}, config=thread_cfg)
            content = getattr(result["messages"][-1], "content", "")
            print(textwrap.fill(_extract_text(content), width=90, subsequent_indent="        ",
                                break_long_words=False, break_on_hyphens=False))
        except Exception as exc:
            print(red(f"\n  [error] {exc}"))
            if os.getenv("DEBUG"): traceback.print_exc()
        print()


async def main():
    if not SERVER_PATH.exists():
        print(red(f"[fatal] server.py not found: {SERVER_PATH}")); sys.exit(1)

    print(dim(f"  Connecting to {SERVER_PATH} ..."))
    mcp_client = MultiServerMCPClient({
        "expense_tracker": {"transport":"stdio","command":sys.executable,"args":[str(SERVER_PATH)]}
    })
    tools = await mcp_client.get_tools()
    print(dim(f"  {len(tools)} tools loaded.\n"))
    graph = await build_graph(tools)
    await run_repl(graph, tools)


if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: pass