"""
app.py
──────
Production-ready Streamlit UI — Luxury Fintech Terminal aesthetic.
All logic identical to original; only CSS/UI layer upgraded.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from datetime import date
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import traceback as _tb

load_dotenv()

st.set_page_config(
    page_title="FinTrack AI",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

SERVER_PATH  = Path(os.getenv("MCP_SERVER_PATH", str(Path(__file__).parent / "server.py")))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GROQ_MODEL   = os.getenv("GROQ_MODEL",   "llama-3.3-70b-versatile")
TODAY        = date.today().isoformat()

QUICK_ACTIONS = [
    ("◈  Cashflow",         "Show me my cashflow for this month"),
    ("◈  Budget status",    "How am I doing against my budgets this month?"),
    ("◈  Top expenses",     "What are my top 10 expenses this month?"),
    ("◈  By category",      "Summarise my spending by category this month"),
    ("◈  Suggest budgets",  "Suggest budgets for next month based on my history"),
    ("◈  Savings",          "How is my savings progress this month?"),
]

SYSTEM_PROMPT = f"""
You are an expert personal finance assistant with full access to a MySQL-backed
expense tracker through a set of MCP tools. Today's date is {TODAY}.

YOUR ROLE: Help users log expenses/income/savings, set budgets, analyse trends.

TOOLS AVAILABLE:
EXPENSE: add_expense, add_expenses_bulk, get_expense_by_id, list_expenses,
  search_expenses, update_expense, delete_expense_by_id, delete_expenses_by_filter, get_categories
ANALYTICS: summarize_by_category, summarize_by_period, get_top_expenses,
  get_cashflow, get_monthly_trend
BUDGET: set_budget, set_budgets_bulk, list_budgets, get_budget_by_id, update_budget,
  delete_budget, get_budget_status, get_budget_vs_actual, get_overbudget_categories,
  get_budget_trend, suggest_budgets_from_history, copy_budgets_from_month
SAVINGS: add_saving, set_savings_goal, get_savings_summary, get_savings_trend

RULES:
1. Always call get_categories before adding expenses.
2. Infer: Food / Transport / Entertainment / Health / Utilities.
3. Use add_expenses_bulk for lists.
4. Currency rupees. today={TODAY}, this month={TODAY[:7]}.
5. Confirm writes with ID + category. Format: rupees 1,200.50.
6. Never hallucinate financial data.
""".strip()

# ── LUXURY FINTECH CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600&family=JetBrains+Mono:wght@300;400;500&family=Syne:wght@300;400;500;600&display=swap');

/* ── Reset & root ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; }

:root {
    --obsidian:    #080a0e;
    --void:        #0b0d13;
    --surface:     #0f1219;
    --surface2:    #141720;
    --surface3:    #1a1e2a;
    --border:      #1e2333;
    --border2:     #262c3d;
    --gold:        #c9a84c;
    --gold2:       #e8c96a;
    --gold-dim:    #8a6f32;
    --gold-glow:   rgba(201,168,76,0.12);
    --gold-glow2:  rgba(201,168,76,0.06);
    --emerald:     #2dd4a0;
    --emerald-dim: rgba(45,212,160,0.15);
    --rose:        #f4617a;
    --rose-dim:    rgba(244,97,122,0.15);
    --amber:       #f59e0b;
    --text-1:      #e8eaf4;
    --text-2:      #9ba3c0;
    --text-3:      #5a6280;
    --text-4:      #363c54;
    --mono:        'JetBrains Mono', monospace;
    --display:     'Playfair Display', serif;
    --sans:        'Syne', sans-serif;
}

/* ── App shell ── */
.stApp {
    background: var(--obsidian);
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(201,168,76,0.04) 0%, transparent 60%),
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='200'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='200' height='200' filter='url(%23n)' opacity='0.025'/%3E%3C/svg%3E");
    font-family: var(--sans);
    color: var(--text-1);
}

html, body, [class*="css"] { font-family: var(--sans); }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--void) !important;
    border-right: 1px solid var(--border) !important;
    padding-top: 0 !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 0;
}
[data-testid="stSidebar"] * { color: var(--text-2) !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: var(--text-1) !important; }

/* ── Main content padding ── */
.main .block-container {
    padding: 2rem 2.5rem 4rem !important;
    max-width: 1000px;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    margin-bottom: 10px !important;
    padding: 16px 20px !important;
    position: relative;
    transition: border-color 0.2s;
}
[data-testid="stChatMessage"]:hover {
    border-color: var(--border2) !important;
}

/* User messages get a gold left bar */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: var(--surface2) !important;
    border-color: var(--border2) !important;
    border-left: 2px solid var(--gold-dim) !important;
}

/* AI messages get emerald left bar */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    border-left: 2px solid rgba(45,212,160,0.3) !important;
}

/* Message text */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span {
    font-family: var(--sans) !important;
    font-size: 14px !important;
    line-height: 1.75 !important;
    color: var(--text-1) !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: var(--surface) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 4px !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--gold-dim) !important;
    box-shadow: 0 0 0 1px var(--gold-dim), 0 0 20px var(--gold-glow) !important;
}
[data-testid="stChatInput"] textarea {
    color: var(--text-1) !important;
    font-family: var(--sans) !important;
    font-size: 14px !important;
    background: transparent !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-3) !important;
}

/* ── Sidebar buttons ── */
.stButton > button {
    background: transparent !important;
    color: var(--text-3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 3px !important;
    font-family: var(--mono) !important;
    font-size: 11px !important;
    font-weight: 400 !important;
    letter-spacing: 0.03em !important;
    padding: 7px 12px !important;
    transition: all 0.15s !important;
    width: 100% !important;
    text-align: left !important;
}
.stButton > button:hover {
    background: var(--gold-glow2) !important;
    border-color: var(--gold-dim) !important;
    color: var(--gold) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    border-radius: 3px !important;
}
[data-testid="stExpander"] summary {
    color: var(--text-3) !important;
    font-family: var(--mono) !important;
    font-size: 11px !important;
    letter-spacing: 0.05em !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 3px !important;
    padding: 12px 14px !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-3) !important;
    font-family: var(--mono) !important;
    font-size: 10px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    color: var(--gold) !important;
    font-family: var(--mono) !important;
    font-size: 20px !important;
    font-weight: 500 !important;
}

/* ── Dividers ── */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 12px 0 !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] > div { border-top-color: var(--gold) !important; }

/* ── Code blocks ── */
[data-testid="stCode"], .stCode {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 3px !important;
    font-family: var(--mono) !important;
    font-size: 12px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--void); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--gold-dim); }

/* ── Tool badge ── */
.tool-badge {
    display: inline-block;
    background: rgba(201,168,76,0.08);
    color: var(--gold);
    border: 1px solid rgba(201,168,76,0.2);
    border-radius: 2px;
    font-family: var(--mono);
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.04em;
    padding: 2px 7px;
    margin: 2px 3px 2px 0;
}

/* ── Status ── */
.status-ok    { color: var(--emerald); font-weight: 500; }
.status-warn  { color: var(--amber); font-weight: 500; }
.status-error { color: var(--rose); font-weight: 500; }

/* ── Sidebar label ── */
.sidebar-label {
    font-family: var(--mono);
    font-size: 9px;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-4) !important;
    margin: 20px 0 8px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
}

/* ── Sidebar brand ── */
.sidebar-brand {
    padding: 24px 16px 20px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 4px;
}
.sidebar-brand-mark {
    font-family: var(--display);
    font-size: 20px;
    font-weight: 500;
    color: var(--gold) !important;
    letter-spacing: -0.3px;
    line-height: 1;
}
.sidebar-brand-sub {
    font-family: var(--mono);
    font-size: 9px;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--text-4) !important;
    margin-top: 4px;
}

/* ── Connection pill ── */
.conn-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 2px;
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.04em;
}
.conn-pill.ok {
    background: var(--emerald-dim);
    border: 1px solid rgba(45,212,160,0.25);
    color: var(--emerald) !important;
}
.conn-pill.err {
    background: var(--rose-dim);
    border: 1px solid rgba(244,97,122,0.25);
    color: var(--rose) !important;
}
.conn-pill.wait {
    background: rgba(245,158,11,0.1);
    border: 1px solid rgba(245,158,11,0.2);
    color: var(--amber) !important;
}
.conn-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: currentColor;
    animation: pulse-dot 2s infinite;
}
@keyframes pulse-dot {
    0%,100% { opacity: 1; }
    50%      { opacity: 0.4; }
}

/* ── Header ── */
.main-header {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    padding-bottom: 20px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 28px;
}
.main-title {
    font-family: var(--display);
    font-size: 28px;
    font-weight: 500;
    color: var(--text-1);
    letter-spacing: -0.5px;
    line-height: 1;
}
.main-title span { color: var(--gold); }
.main-date {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text-4);
    letter-spacing: 0.06em;
    padding-bottom: 3px;
}

/* ── Thinking animation ── */
.thinking-wrap {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 4px 0;
}
.thinking-label {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--gold-dim);
    letter-spacing: 0.06em;
}
.thinking-bars {
    display: flex;
    gap: 3px;
    align-items: flex-end;
    height: 14px;
}
.thinking-bars span {
    display: block;
    width: 2px;
    background: var(--gold);
    border-radius: 1px;
    animation: bar-bounce 1s ease-in-out infinite;
}
.thinking-bars span:nth-child(1) { height: 4px;  animation-delay: 0s; }
.thinking-bars span:nth-child(2) { height: 10px; animation-delay: 0.1s; }
.thinking-bars span:nth-child(3) { height: 7px;  animation-delay: 0.2s; }
.thinking-bars span:nth-child(4) { height: 12px; animation-delay: 0.3s; }
.thinking-bars span:nth-child(5) { height: 5px;  animation-delay: 0.4s; }
@keyframes bar-bounce {
    0%,100% { transform: scaleY(0.4); opacity: 0.4; }
    50%      { transform: scaleY(1);   opacity: 1; }
}

/* ── Welcome card ── */
.welcome-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-top: 2px solid var(--gold-dim);
    border-radius: 4px;
    padding: 28px 32px;
    margin-bottom: 24px;
}
.welcome-title {
    font-family: var(--display);
    font-size: 18px;
    font-weight: 500;
    color: var(--text-1);
    margin-bottom: 8px;
}
.welcome-sub {
    font-family: var(--sans);
    font-size: 13px;
    color: var(--text-3);
    line-height: 1.6;
    margin-bottom: 20px;
}
.cap-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
}
.cap-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 10px 12px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 3px;
}
.cap-icon {
    font-family: var(--mono);
    font-size: 12px;
    color: var(--gold);
    margin-top: 1px;
    flex-shrink: 0;
}
.cap-text {
    font-family: var(--sans);
    font-size: 12px;
    color: var(--text-2);
    line-height: 1.4;
}
.cap-text strong { color: var(--text-1); font-weight: 500; }

/* ── Tool call info line ── */
.tool-line {
    margin-top: 10px;
    padding-top: 8px;
    border-top: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text-4);
    letter-spacing: 0.04em;
}

/* ── Info/error boxes ── */
[data-testid="stAlert"] {
    border-radius: 3px !important;
    font-family: var(--sans) !important;
    font-size: 13px !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────

def _init_state():
    defaults = {
        "thread_id":  str(uuid.uuid4()),
        "messages":   [],
        "tool_log":   [],
        "graph":      None,
        "tools":      [],
        "mcp_ready":  False,
        "mcp_client": None,
        "turn_count": 0,
        "error":      None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────
# LLM + GRAPH
# ─────────────────────────────────────────────

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

QUOTA_MARKERS = ["quota","rate limit","resource exhausted","429","too many requests","quota exceeded"]

def _is_quota_error(exc):
    return any(m in str(exc).lower() for m in QUOTA_MARKERS)

class FallbackLLM:
    def __init__(self, primary, fallback):
        self._primary = primary; self._fallback = fallback
        self._using_fallback = False

    def bind_tools(self, tools, **kwargs):
        b = FallbackLLM(
            primary=self._primary.bind_tools(tools, **kwargs),
            fallback=self._fallback.bind_tools(tools, **kwargs),
        )
        b._using_fallback = self._using_fallback
        return b

    async def ainvoke(self, messages, config=None, **kwargs):
        if self._using_fallback:
            return await self._fallback.ainvoke(messages, config=config, **kwargs)
        try:
            return await self._primary.ainvoke(messages, config=config, **kwargs)
        except Exception as exc:
            if _is_quota_error(exc):
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
    gk = os.getenv("GEMINI_API_KEY")
    qk = os.getenv("GROQ_API_KEY")
    if not gk: raise RuntimeError("GEMINI_API_KEY not set.")
    if not qk: raise RuntimeError("GROQ_API_KEY not set.")
    return FallbackLLM(
        primary=ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0, google_api_key=gk),
        fallback=ChatGroq(model=GROQ_MODEL, temperature=0, groq_api_key=qk),
    )

def _build_graph(tools):
    llm = _build_llm()
    llm_with_tools = llm.bind_tools(tools)

    async def chat_node(state):
        msgs = state["messages"]
        if not msgs or not isinstance(msgs[0], SystemMessage):
            msgs = [SystemMessage(content=SYSTEM_PROMPT)] + list(msgs)
        return {"messages": [await llm_with_tools.ainvoke(msgs)]}

    tool_node = ToolNode(tools)
    g = StateGraph(ChatState)
    g.add_node("chat_node", chat_node)
    g.add_node("tool_node", tool_node)
    g.add_edge(START, "chat_node")
    g.add_conditional_edges("chat_node", tools_condition, {"tools": "tool_node", END: END})
    g.add_edge("tool_node", "chat_node")
    return g.compile(checkpointer=InMemorySaver())


# ─────────────────────────────────────────────
# ASYNC BRIDGE
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _get_event_loop():
    import threading
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    return loop

def _run(coro):
    import concurrent.futures
    return asyncio.run_coroutine_threadsafe(coro, _get_event_loop()).result(timeout=60)


# ─────────────────────────────────────────────
# MCP INIT
# ─────────────────────────────────────────────

async def _init_mcp():
    client = MultiServerMCPClient({
        "expense_tracker": {
            "transport": "stdio",
            "command":   sys.executable,
            "args":      [str(SERVER_PATH)],
        }
    })
    tools = await client.get_tools()
    graph = _build_graph(tools)
    return client, tools, graph

def ensure_mcp():
    if st.session_state.mcp_ready: return True
    if not SERVER_PATH.exists():
        st.session_state.error = f"server.py not found: {SERVER_PATH}"; return False
    try:
        client, tools, graph = _run(_init_mcp())
        st.session_state.mcp_client = client
        st.session_state.tools      = tools
        st.session_state.graph      = graph
        st.session_state.mcp_ready  = True
        return True
    except Exception as exc:
        st.session_state.error = "".join(_tb.format_exception(type(exc), exc, exc.__traceback__))
        return False


# ─────────────────────────────────────────────
# AGENT INVOKE
# ─────────────────────────────────────────────

def _extract_text(content):
    if isinstance(content, str): return content.strip()
    if isinstance(content, list):
        parts = [b.get("text","") if isinstance(b,dict) else str(b) for b in content]
        return "\n".join(p for p in parts if p).strip()
    return str(content).strip()

def invoke_agent(user_text):
    graph     = st.session_state.graph
    thread_id = st.session_state.thread_id
    cfg       = {"configurable": {"thread_id": thread_id}}

    async def _run_agent():
        return await graph.ainvoke({"messages": [HumanMessage(content=user_text)]}, config=cfg)

    result   = _run(_run_agent())
    messages = result.get("messages", [])

    tools_called = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                tools_called.append(tc.get("name", "unknown"))

    reply = ""
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content:
            reply = _extract_text(m.content)
            if reply: break

    return reply, tools_called


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        # Brand
        st.markdown("""
        <div class='sidebar-brand'>
            <div class='sidebar-brand-mark'>◈ FinTrack</div>
            <div class='sidebar-brand-sub'>AI · Personal Finance Terminal</div>
        </div>
        """, unsafe_allow_html=True)

        # Connection
        st.markdown("<div class='sidebar-label'>System</div>", unsafe_allow_html=True)
        if st.session_state.mcp_ready:
            st.markdown(
                f"<div class='conn-pill ok'><span class='conn-dot'></span>LIVE · MCP connected</div>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div style='font-family:var(--mono);font-size:10px;color:var(--text-4);margin-top:6px;padding-left:2px'>"
                f"PRIMARY  {GEMINI_MODEL}<br>"
                f"FALLBACK {GROQ_MODEL}<br>"
                f"TOOLS    {len(st.session_state.tools)} loaded</div>",
                unsafe_allow_html=True
            )
        elif st.session_state.error:
            st.markdown("<div class='conn-pill err'><span class='conn-dot'></span>ERROR</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='conn-pill wait'><span class='conn-dot'></span>CONNECTING</div>", unsafe_allow_html=True)

        st.divider()

        # Session stats
        st.markdown("<div class='sidebar-label'>Session</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("TURNS", st.session_state.turn_count)
        c2.metric("TOOLS", len(st.session_state.tool_log))

        st.divider()

        # Quick actions
        st.markdown("<div class='sidebar-label'>Quick actions</div>", unsafe_allow_html=True)
        for label, query in QUICK_ACTIONS:
            if st.button(label, key=f"qa_{label}"):
                st.session_state["_pending_query"] = query
                st.rerun()

        st.divider()

        # Tool log
        st.markdown("<div class='sidebar-label'>Tool log</div>", unsafe_allow_html=True)
        if st.session_state.tool_log:
            for turn_num, tools_used in reversed(st.session_state.tool_log[-10:]):
                badges = "".join(f"<span class='tool-badge'>{t}</span>" for t in tools_used)
                st.markdown(
                    f"<div style='margin-bottom:8px'>"
                    f"<div style='font-family:var(--mono);font-size:9px;color:var(--text-4);margin-bottom:3px'>T{turn_num:02d}</div>"
                    f"{badges}</div>",
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                "<div style='font-family:var(--mono);font-size:10px;color:var(--text-4)'>— no activity yet —</div>",
                unsafe_allow_html=True
            )

        st.divider()

        if st.button("⊘  Clear conversation", key="new_convo"):
            st.session_state.thread_id  = str(uuid.uuid4())
            st.session_state.messages   = []
            st.session_state.tool_log   = []
            st.session_state.turn_count = 0
            st.rerun()

        if st.session_state.tools:
            with st.expander("◈  All tools"):
                for t in st.session_state.tools:
                    name = getattr(t, "name", str(t))
                    st.markdown(f"<span class='tool-badge'>{name}</span>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CHAT HISTORY
# ─────────────────────────────────────────────

def render_messages():
    for msg in st.session_state.messages:
        role    = msg["role"]
        content = msg["content"]
        tools   = msg.get("tools", [])
        # FIX: Use valid emoji avatars — "◈" is not a valid Streamlit avatar
        avatar  = "🧑" if role == "user" else "💎"

        with st.chat_message(role, avatar=avatar):
            st.markdown(content)
            if tools:
                badges = "".join(f"<span class='tool-badge'>{t}</span>" for t in tools)
                st.markdown(
                    f"<div class='tool-line'>called → {badges}</div>",
                    unsafe_allow_html=True
                )


# ─────────────────────────────────────────────
# BOOTSTRAP
# ─────────────────────────────────────────────

if not st.session_state.mcp_ready:
    with st.spinner("Initialising FinTrack AI…"):
        ensure_mcp()
    if st.session_state.mcp_ready:
        st.rerun()

render_sidebar()

# ── Header ──────────────────────────────────
st.markdown(f"""
<div class='main-header'>
    <div>
        <div class='main-title'>Finance <span>Terminal</span></div>
    </div>
    <div class='main-date'>{TODAY} &nbsp;·&nbsp; AI-POWERED</div>
</div>
""", unsafe_allow_html=True)

# ── Error ────────────────────────────────────
if st.session_state.error and not st.session_state.mcp_ready:
    st.error("**Startup failure** — see trace below")
    st.code(st.session_state.error, language="text")
    st.info(
        f"Checklist:\n"
        f"- server.py → `{SERVER_PATH}`\n"
        f"- MySQL running + credentials in `.env`\n"
        f"- `pip install fastmcp aiomysql langchain-groq langchain-google-genai`\n"
        f"- GEMINI_API_KEY + GROQ_API_KEY set in `.env`"
    )
    if st.button("↺  Retry"):
        st.session_state.mcp_ready = False
        st.session_state.error     = None
        st.rerun()
    st.stop()

# ── Welcome ──────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class='welcome-card'>
        <div class='welcome-title'>Good to have you back.</div>
        <div class='welcome-sub'>Your personal finance intelligence layer. Ask naturally, or use the quick actions.</div>
        <div class='cap-grid'>
            <div class='cap-item'>
                <div class='cap-icon'>$</div>
                <div class='cap-text'><strong>Log transactions</strong><br>Expenses, income, savings</div>
            </div>
            <div class='cap-item'>
                <div class='cap-icon'>≋</div>
                <div class='cap-text'><strong>Track budgets</strong><br>Set limits, monitor pace</div>
            </div>
            <div class='cap-item'>
                <div class='cap-icon'>↗</div>
                <div class='cap-text'><strong>Analyse trends</strong><br>Category, period, cashflow</div>
            </div>
            <div class='cap-item'>
                <div class='cap-icon'>◎</div>
                <div class='cap-text'><strong>Smart suggestions</strong><br>AI-powered budget planning</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Messages ─────────────────────────────────
render_messages()

# ── Input ────────────────────────────────────
pending    = st.session_state.pop("_pending_query", None)
user_input = st.chat_input("Ask anything about your finances…") or pending

if user_input and st.session_state.mcp_ready:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    # FIX: Use valid emoji avatar "💎" instead of "◈"
    with st.chat_message("assistant", avatar="💎"):
        thinking = st.empty()
        thinking.markdown("""
        <div class='thinking-wrap'>
            <div class='thinking-bars'>
                <span></span><span></span><span></span><span></span><span></span>
            </div>
            <div class='thinking-label'>processing</div>
        </div>
        """, unsafe_allow_html=True)

        try:
            reply, tools_called = invoke_agent(user_input)
            thinking.empty()
            st.markdown(reply)
            if tools_called:
                badges = "".join(f"<span class='tool-badge'>{t}</span>" for t in tools_called)
                st.markdown(
                    f"<div class='tool-line'>called → {badges}</div>",
                    unsafe_allow_html=True
                )
        except Exception as exc:
            thinking.empty()
            reply        = f"Error: {exc}"
            tools_called = []
            st.error(reply)

    st.session_state.messages.append({
        "role":    "assistant",
        "content": reply,
        "tools":   tools_called,
    })
    st.session_state.turn_count += 1
    if tools_called:
        st.session_state.tool_log.append((st.session_state.turn_count, tools_called))

    st.rerun()