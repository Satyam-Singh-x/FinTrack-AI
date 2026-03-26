"""
app.py  —  FinTrack
────────────────────
Production-ready Streamlit UI for the FinTrack MCP client.

LLM Chain: Gemini 2.0 Flash  →  Ollama (local fallback)

Usage
─────
  streamlit run app.py
  GEMINI_API_KEY=... streamlit run app.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from datetime import date, timedelta
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

load_dotenv()

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="FinTrack — AI Finance",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

SERVER_PATH  = Path(os.getenv("MCP_SERVER_PATH", str(Path(__file__).parent / "server.py")))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://localhost:11434")
TODAY        = date.today().isoformat()
THIS_MONTH   = date.today().strftime("%Y-%m")
_last_day    = (date.today().replace(day=1) - timedelta(days=1))
LAST_MONTH   = _last_day.strftime("%Y-%m")
WEEK_START   = (date.today() - timedelta(days=date.today().weekday())).isoformat()

QUICK_ACTIONS = [
    ("Cashflow overview",    "Show me my cashflow for this month"),
    ("Budget status",        "How am I doing against my budgets this month?"),
    ("Top 10 expenses",      "What are my top 10 expenses this month?"),
    ("Spending by category", "Summarise my spending by category this month"),
    ("Suggest budgets",      "Suggest budgets for next month based on my history"),
    ("Savings progress",     "How is my savings progress this month?"),
]

TIER_LABELS  = ["Gemini", "Ollama"]
TIER_COLOURS = ["tier-gemini", "tier-ollama"]

# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────
SYSTEM_PROMPT = f"""
You are FinTrack AI — an expert personal finance assistant and Chartered Accountant (CA) with deep knowledge of personal budgeting, expense management, cash flow analysis, and financial planning. You have full access to a MySQL-backed expense tracker through MCP tools.

Today's date is {TODAY}. Current month is {TODAY[:7]}.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IDENTITY & EXPERTISE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You think like a CA. You don't just log numbers — you interpret them.
- You notice when spending is accelerating mid-month and warn the user
- You spot categories that consistently exceed budget and suggest adjustments
- You calculate savings rates and compare them to the 50/30/20 rule
- You identify top expense categories and suggest where to cut
- You proactively offer insights the user didn't ask for but would want to know
- You speak in plain language, not financial jargon, unless the user prefers it
- You always provide context: "₹3,200 on Food this month — that's 18% above your usual average"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AVAILABLE TOOLS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXPENSE MANAGEMENT
  add_expense            — log one transaction (expense / income / saving)
  add_expenses_bulk      — log multiple transactions in one call (ALWAYS prefer for lists)
  get_expense_by_id      — fetch a single transaction by ID
  list_expenses          — paginated, filterable transaction list
  search_expenses        — full-text search across notes, tags, category
  update_expense         — edit any field of an existing entry by ID
  delete_expense_by_id   — delete one transaction
  delete_expenses_by_filter — bulk delete with date/type/category filters
  get_categories         — all distinct category + subcategory pairs ever used

ANALYTICS & REPORTING
  summarize_by_category  — totals grouped by category for a date range
  summarize_by_period    — daily / weekly / monthly spending buckets
  get_top_expenses       — top-N largest single transactions
  get_cashflow           — income vs expense vs saving, net position, savings rate
  get_monthly_trend      — month-over-month trend for a specific category

BUDGET MANAGEMENT
  set_budget             — create or update a monthly spending limit for a category
  set_budgets_bulk       — upsert multiple budgets in one call
  list_budgets           — all budgets or savings goals for a month
  get_budget_by_id       — fetch one budget row by ID
  update_budget          — change amount or note of an existing budget
  delete_budget          — remove by ID or month + category

BUDGET ANALYTICS
  get_budget_status      — live actual vs budget, pace label, status label per category
  get_budget_vs_actual   — full side-by-side month review including unbudgeted categories
  get_overbudget_categories — only categories that crossed a usage threshold %
  get_budget_trend       — month-over-month budget vs actual for one category
  suggest_budgets_from_history — AI-suggested limits based on past average spend
  copy_budgets_from_month — carry all budgets forward from one month to another

SAVINGS
  add_saving             — log a conscious saving event with amount and context
  set_savings_goal       — set a monthly savings target per category
  get_savings_summary    — actual savings vs goals for a month with pace tracking
  get_savings_trend      — month-over-month savings history

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT BEHAVIOURAL RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RULE 1 — ALWAYS CALL get_categories FIRST BEFORE ANY ADD/UPDATE
  Before calling add_expense, add_expenses_bulk, or update_expense — always call
  get_categories first to retrieve existing category and subcategory names.
  Match the user's input to the closest existing category (case-sensitive exact match).
  Only create a new category name if there is genuinely no matching existing one.
  This prevents duplicates like "food", "Food", "Foods", "FOOD" all existing separately.

RULE 2 — CATEGORY INFERENCE HIERARCHY
  Use this priority order to assign categories:
  1. Exact match from get_categories results
  2. Fuzzy match (e.g. user says "zomato" → existing "Food" category)
  3. Infer from context using these mappings:
     - restaurant / cafe / hotel / lunch / dinner / breakfast / snack /
       groceries / supermarket / zomato / swiggy / coffee → Food
     - uber / ola / auto / rickshaw / taxi / metro / bus / petrol /
       diesel / fuel / toll / parking / cab → Transport
     - netflix / prime / hotstar / spotify / movies / concert / game /
       cricket / bar / pub / outing / weekend → Entertainment
     - doctor / hospital / medicine / pharmacy / gym / yoga /
       health insurance / clinic / chemist → Health
     - electricity / water / gas / internet / broadband / wifi /
       phone bill / rent / maintenance / society → Utilities
     - school / college / course / books / tuition / coaching → Education
     - flight / hotel / trip / holiday / travel / resort → Travel
     - emi / loan / credit card / insurance premium → Finance
     - amazon / flipkart / myntra / shopping / clothes / electronics → Shopping
  4. If still ambiguous — ask the user ONE specific question

RULE 3 — BULK OPERATIONS MANDATORY FOR LISTS
  If the user mentions 2 or more transactions in a single message,
  you MUST use add_expenses_bulk — never call add_expense multiple times.
  This is non-negotiable for performance and atomicity.

RULE 4 — DATE RESOLUTION (NEVER FABRICATE)
  - "today"           → {TODAY}
  - "yesterday"       → calculate one day before {TODAY}
  - "this week"       → Monday of current week to {TODAY}
  - "last week"       → full Monday–Sunday of previous week
  - "this month"      → {TODAY[:7]}-01 to {TODAY}
  - "last month"      → first to last day of the month before {TODAY[:7]}
  - "this year"       → {TODAY[:4]}-01-01 to {TODAY}
  - Never assume or fabricate a date — derive it mathematically from {TODAY}

RULE 5 — CURRENCY & NUMBER FORMATTING
  - Always use ₹ symbol for Indian Rupees
  - Format: ₹1,200.50 — never 1200.5 or Rs 1200 or INR 1200
  - Never round amounts — store and display exact values
  - For large numbers: ₹1,20,000 (Indian numbering) or ₹1.2L for casual mention
  - Percentages: always include % symbol, round to 2 decimal places

RULE 6 — WRITE OPERATION CONFIRMATIONS
  After every successful add / update / delete, confirm with:
  - What was stored (amount, category, date)
  - The database ID assigned
  - Any relevant insight (e.g. "This brings your Food total to ₹2,800 this month")

RULE 7 — ANALYTICS INTELLIGENCE
  Never just dump raw data. Always interpret it:
  - "How am I doing?" → call get_budget_status AND get_cashflow, then synthesise
  - "Where did my money go?" → call summarize_by_category, highlight top 3
  - "Am I saving enough?" → call get_cashflow, calculate savings rate,
    compare to 20% rule (50% needs / 30% wants / 20% savings)
  - For any trend query → call get_monthly_trend and comment on direction
  - If budget is over 85% used before month end → proactively warn
  - If net cashflow is negative → flag it clearly as a priority concern

RULE 8 — PROACTIVE CA INSIGHTS
  Volunteer insights even when not asked:
  - After logging an expense that pushes a category near its budget limit → warn
  - After get_cashflow → comment on savings rate vs recommended 20%
  - After summarize_by_category → identify the single biggest opportunity to cut
  - After get_budget_status → call out CRITICAL or OVER BUDGET categories first
  - Use plain language: "You've spent ₹4,200 on Food with 12 days left —
    at this pace you'll overshoot your ₹5,000 budget by about ₹800"

RULE 9 — RESPONSE FORMAT STANDARDS
  - Lead with the key insight or confirmation, not raw data
  - Use bullet points for lists of 3+ items
  - Use a simple table format for budget comparisons (category | budget | actual | %)
  - Keep responses under 200 words unless the user asks for full detail
  - If data is large (>10 rows), summarise first and offer: "Want the full breakdown?"
  - Never show raw JSON or database IDs in the main response (IDs only in confirmation)

RULE 10 — DATA INTEGRITY
  - Never hallucinate financial figures — only report what tools return
  - If a tool returns empty results, say so clearly and suggest next steps
  - If the user asks about a period with no data, explain and offer to log some
  - Never estimate or approximate stored values — always fetch from the database

RULE 11 — TOOL CALL FORMAT (CRITICAL)
  When calling a tool, return ONLY clean JSON arguments. Example:
  CORRECT:   {{"month": "2025-03", "category": "Food"}}
  INCORRECT: {{"function": "get_budget_status", "args": {{"month": "2025-03"}}}}
  - Never wrap arguments in "function" or "args" keys
  - Never nest the arguments — flat JSON only
  - Call one tool at a time and wait for the result before deciding the next step
  - Never skip required fields — check tool signatures carefully

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FINANCIAL KNOWLEDGE BASE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Apply these principles when analysing a user's finances:

50/30/20 RULE
  50% of take-home income → Needs (rent, groceries, utilities, EMIs)
  30% of take-home income → Wants (dining out, entertainment, shopping)
  20% of take-home income → Savings and investments

EMERGENCY FUND BENCHMARK
  Ideal: 6 months of monthly expenses set aside as liquid savings
  Minimum: 3 months for salaried individuals

SAVINGS RATE BENCHMARKS
  < 10%  → Concerning — flag and suggest specific cuts
  10–20% → Average — encourage improvement
  20–30% → Good — acknowledge and suggest investment options
  > 30%  → Excellent — commend and suggest growth strategies

BUDGET HEALTH LABELS (use these in responses)
  HEALTHY   → < 60% of budget used
  ON TRACK  → 60–84% used
  CRITICAL  → 85–99% used — warn proactively
  OVER BUDGET → 100%+ used — flag immediately with overspend amount

PACE ANALYSIS
  When days_elapsed / days_in_month < pct_budget_used → OVERSPENDING
  When they are roughly equal → ON PACE
  When days_elapsed / days_in_month > pct_budget_used → UNDERSPENDING (good)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONVERSATION STYLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- Be direct and confident like a CA who knows what they're talking about
- Be warm but not sycophantic — skip "Great question!" and similar filler
- Use "you" language, not "the user" — speak directly to the person
- When something is financially concerning, say so clearly without sugarcoating
- When the user is doing well, acknowledge it genuinely and specifically
- Remember context from earlier in the conversation and refer back to it
- If asked something outside your scope, say so and redirect to what you can help with
""".strip()

# ─────────────────────────────────────────────
# CUSTOM CSS  —  FinTrack Premium Dark Theme
# ─────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@300;400;500&family=Instrument+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap');

/* ── Reset & Base ─────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Instrument Sans', sans-serif;
    letter-spacing: -0.01em;
}

.stApp {
    background: #080a0f;
    color: #dde1ec;
}

/* ── Animated gradient background ─────────── */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(99,102,241,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(16,185,129,0.06) 0%, transparent 55%),
        radial-gradient(ellipse 40% 60% at 50% 50%, rgba(245,158,11,0.02) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ───────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0c0e15 !important;
    border-right: 1px solid rgba(255,255,255,0.05) !important;
}
[data-testid="stSidebar"] * {
    color: #8891aa !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #eef0f8 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
}

/* ── Chat messages ─────────────────────────── */
[data-testid="stChatMessage"] {
    border-radius: 16px;
    margin-bottom: 10px;
    padding: 16px 20px !important;
    border: 1px solid rgba(255,255,255,0.05);
    background: rgba(255,255,255,0.025);
    backdrop-filter: blur(12px);
    transition: border-color 0.2s ease;
}
[data-testid="stChatMessage"]:hover {
    border-color: rgba(255,255,255,0.08);
}

/* ── Chat input ────────────────────────────── */
[data-testid="stChatInput"] > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 14px !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
[data-testid="stChatInput"] > div:focus-within {
    border-color: rgba(99,102,241,0.5) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
}
[data-testid="stChatInput"] textarea {
    color: #eef0f8 !important;
    font-family: 'Instrument Sans', sans-serif !important;
    font-size: 14px !important;
}

/* ── Quick action buttons ──────────────────── */
.stButton > button {
    background: rgba(255,255,255,0.03) !important;
    color: #8891aa !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
    font-family: 'Instrument Sans', sans-serif !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 8px 12px !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
    text-align: left !important;
    letter-spacing: 0 !important;
}
.stButton > button:hover {
    background: rgba(99,102,241,0.1) !important;
    border-color: rgba(99,102,241,0.3) !important;
    color: #c4c8e8 !important;
    transform: translateX(2px) !important;
}

/* ── Expander ──────────────────────────────── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
    color: #5a6080 !important;
    font-size: 12px !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* ── Metrics ───────────────────────────────── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 14px !important;
}
[data-testid="stMetricLabel"] {
    color: #5a6080 !important;
    font-size: 11px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
[data-testid="stMetricValue"] {
    color: #eef0f8 !important;
    font-size: 24px !important;
    font-weight: 700 !important;
    font-family: 'Syne', sans-serif !important;
}

hr { border-color: rgba(255,255,255,0.05) !important; }

/* ── Tool badge ────────────────────────────── */
.tool-badge {
    display: inline-block;
    background: rgba(99,102,241,0.1);
    color: #818cf8;
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    font-weight: 500;
    padding: 2px 7px;
    margin: 2px 2px 2px 0;
    letter-spacing: 0.02em;
}

/* ── LLM tier badges ───────────────────────── */
.tier-gemini {
    background: rgba(59,130,246,0.1) !important;
    color: #60a5fa !important;
    border-color: rgba(59,130,246,0.2) !important;
}
.tier-ollama {
    background: rgba(16,185,129,0.1) !important;
    color: #34d399 !important;
    border-color: rgba(16,185,129,0.2) !important;
}

/* ── Status classes ────────────────────────── */
.status-ok    { color: #34d399; font-weight: 500; }
.status-warn  { color: #fbbf24; font-weight: 500; }
.status-error { color: #f87171; font-weight: 500; }

/* ── App header ────────────────────────────── */
.app-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 0 0 20px 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 24px;
}
.fintrack-logo {
    width: 38px;
    height: 38px;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #10b981 100%);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    color: white;
    flex-shrink: 0;
    box-shadow: 0 4px 20px rgba(79,70,229,0.35);
    letter-spacing: -0.05em;
}
.app-title {
    font-size: 20px;
    font-weight: 800;
    font-family: 'Syne', sans-serif;
    color: #eef0f8;
    letter-spacing: -0.04em;
    line-height: 1.1;
}
.app-title span {
    background: linear-gradient(90deg, #818cf8, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.app-subtitle {
    font-size: 12px;
    color: #4a5070;
    margin-top: 3px;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.02em;
}
.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #34d399;
    box-shadow: 0 0 8px rgba(52,211,153,0.6);
    flex-shrink: 0;
    animation: pulse-dot 2.5s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%, 100% { box-shadow: 0 0 6px rgba(52,211,153,0.5); }
    50%       { box-shadow: 0 0 14px rgba(52,211,153,0.9); }
}

/* ── Sidebar section headers ───────────────── */
.sidebar-section {
    font-size: 10px;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #2e3450 !important;
    margin: 18px 0 8px 0;
}

/* ── Sidebar brand ─────────────────────────── */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 4px 0 12px 0;
}
.sidebar-brand-logo {
    width: 30px;
    height: 30px;
    background: linear-gradient(135deg, #4f46e5, #10b981);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 13px;
    color: white;
    letter-spacing: -0.04em;
    flex-shrink: 0;
}
.sidebar-brand-name {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 17px !important;
    color: #eef0f8 !important;
    letter-spacing: -0.04em !important;
}
.sidebar-brand-tagline {
    font-size: 10px !important;
    color: #3a4060 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    margin-top: 1px;
}

/* ── Thinking animation ────────────────────── */
.thinking-container {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 4px 0;
}
.thinking-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: #4a5070;
    letter-spacing: 0.04em;
}
.thinking-dots span {
    display: inline-block;
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: #4f46e5;
    margin: 0 2px;
    animation: blink 1.4s infinite ease-in-out;
}
.thinking-dots span:nth-child(1) { animation-delay: 0s; }
.thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
.thinking-dots span:nth-child(3) { animation-delay: 0.4s; }
@keyframes blink {
    0%, 80%, 100% { opacity: 0.15; transform: scale(0.8); }
    40%           { opacity: 1;    transform: scale(1.2); }
}

/* ── Divider variant ───────────────────────── */
.subtle-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.04);
    margin: 12px 0;
}

/* ── Tool called row ───────────────────────── */
.tools-called-row {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid rgba(255,255,255,0.04);
}
.tools-called-label {
    font-size: 10px;
    font-family: 'IBM Plex Mono', monospace;
    color: #2e3450;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    flex-shrink: 0;
}

/* ── New conversation button special ──────── */
[data-testid="stSidebar"] [data-testid="stButton"]:last-of-type > button {
    background: rgba(239,68,68,0.07) !important;
    border-color: rgba(239,68,68,0.15) !important;
    color: #f87171 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.02em !important;
}
[data-testid="stSidebar"] [data-testid="stButton"]:last-of-type > button:hover {
    background: rgba(239,68,68,0.14) !important;
    border-color: rgba(239,68,68,0.3) !important;
    color: #fca5a5 !important;
    transform: none !important;
}

/* ── Welcome card ──────────────────────────── */
.welcome-card {
    background: linear-gradient(135deg, rgba(79,70,229,0.08) 0%, rgba(16,185,129,0.05) 100%);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 8px;
}
.welcome-title {
    font-family: 'Syne', sans-serif;
    font-size: 16px;
    font-weight: 700;
    color: #eef0f8;
    margin-bottom: 6px;
    letter-spacing: -0.03em;
}
.welcome-sub {
    font-size: 13px;
    color: #5a6080;
    line-height: 1.6;
}
.capability-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 14px;
}
.cap-pill {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 11px;
    color: #8891aa;
    font-family: 'Instrument Sans', sans-serif;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────

def _init_state():
    defaults = {
        "thread_id":    str(uuid.uuid4()),
        "messages":     [],
        "tool_log":     [],
        "graph":        None,
        "tools":        [],
        "mcp_ready":    False,
        "mcp_client":   None,
        "turn_count":   0,
        "error":        None,
        "active_tier":  0,
        "_llm_ref":     None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────
# CONTENT EXTRACTION
# ─────────────────────────────────────────────

def _extract_text(content) -> str:
    """
    Safely extract a plain string from any LLM message content shape:
      - str                                   → plain str response
      - list[{"type":"text","text": ...}]     → Gemini multi-part
      - list[str]                             → rare fallback
    """
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p for p in parts if p).strip()
    return str(content).strip()


# ─────────────────────────────────────────────
# QUOTA / FALLBACK HELPERS
# ─────────────────────────────────────────────

QUOTA_MARKERS = [
    "quota", "rate limit", "resource exhausted", "429",
    "too many requests", "quota exceeded", "rate_limit_exceeded",
    "insufficient_quota", "billing", "exceeded your",
]

def _is_quota_error(exc: Exception) -> bool:
    return any(m in str(exc).lower() for m in QUOTA_MARKERS)


# ─────────────────────────────────────────────
# TWO-TIER FALLBACK LLM  (Gemini → Ollama)
# ─────────────────────────────────────────────

class FallbackLLM:
    """
    Chain: Gemini 2.0 Flash  →  Ollama (local).
    Advances automatically on any quota / rate-limit exception.
    """

    def __init__(self, models: list):
        if not models:
            raise ValueError("Need at least one model.")
        self._models = models
        self._tier   = 0

    @property
    def _active(self):
        return self._models[self._tier]

    def _label(self, idx: int | None = None) -> str:
        i = self._tier if idx is None else idx
        return TIER_LABELS[i] if i < len(TIER_LABELS) else f"model-{i}"

    def _advance(self, exc: Exception) -> bool:
        if self._tier < len(self._models) - 1:
            prev = self._label()
            self._tier += 1
            try:
                st.session_state.active_tier = self._tier
            except Exception:
                pass
            st.toast(
                f"⚠️ {prev} quota hit — switching to local **{self._label()}**",
                icon="🔄",
            )
            return True
        st.toast("❌ All LLM tiers exhausted.", icon="🚨")
        return False

    def bind_tools(self, tools, **kwargs):
        bound_models = []
        for m in self._models:
            try:
                bound_models.append(m.bind_tools(tools, **kwargs))
            except Exception:
                try:
                    bound_models.append(m.bind_tools(tools))
                except Exception:
                    bound_models.append(m)
        new = FallbackLLM(bound_models)
        new._tier = self._tier
        return new

    async def ainvoke(self, messages, config=None, **kwargs):
        while True:
            try:
                return await self._active.ainvoke(messages, config=config, **kwargs)
            except Exception as exc:
                if _is_quota_error(exc) and self._advance(exc):
                    continue
                raise

    def invoke(self, messages, config=None, **kwargs):
        while True:
            try:
                return self._active.invoke(messages, config=config, **kwargs)
            except Exception as exc:
                if _is_quota_error(exc) and self._advance(exc):
                    continue
                raise


# ─────────────────────────────────────────────
# LLM FACTORY  (Gemini + Ollama only)
# ─────────────────────────────────────────────

def _build_llm() -> FallbackLLM:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_ollama       import ChatOllama

    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Get a free key at https://aistudio.google.com"
        )

    gemini = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0,
        google_api_key=gemini_key,
    )

    # Ollama local fallback — format="json" for tool-call support on smaller models
    ollama = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_URL,
        temperature=0,
        format="json",
    )

    return FallbackLLM(models=[gemini, ollama])


# ─────────────────────────────────────────────
# GRAPH WIRING
# ─────────────────────────────────────────────

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _sanitize_tool_calls(message):
    """
    Gemini sometimes embeds the function name as a key inside the args dict:
      {'get_budget_status': {'month': '2023-03'}}   ← broken
    or passes args as a nested dict with 'function'/'args' keys.
    This normalises all tool_calls to plain {param: value} dicts.
    """
    if not isinstance(message, AIMessage) or not getattr(message, "tool_calls", None):
        return message

    clean_calls = []
    for tc in message.tool_calls:
        args = tc.get("args", {})
        name = tc.get("name", "")

        # Pattern 1: {'function_name': {...actual args...}}
        if isinstance(args, dict) and len(args) == 1:
            only_key = next(iter(args))
            only_val = args[only_key]
            if isinstance(only_val, dict) and only_key == name:
                args = only_val

        # Pattern 2: {'function': 'name', 'args': {...}} or {'function': 'name', ...}
        if isinstance(args, dict) and "function" in args:
            inner = args.get("args", None)
            if isinstance(inner, dict):
                args = inner
            else:
                args = {k: v for k, v in args.items() if k != "function"}

        # Pattern 3: outer dict wraps everything under the tool name as a key
        if isinstance(args, dict) and name in args and isinstance(args[name], dict):
            args = args[name]

        clean_calls.append({**tc, "args": args})

    # Rebuild the AIMessage with sanitized tool_calls
    new_msg = AIMessage(
        content=message.content,
        tool_calls=clean_calls,
        id=getattr(message, "id", None),
    )
    return new_msg


def _fix_tool_call_args(tool_call: dict, tool_name: str) -> dict:
    """
    Gemini sometimes double-wraps tool arguments, e.g.:
      {"get_budget_status": {"month": "2026-03"}}   <- wrong (double-wrapped)
      {"month": "2026-03"}                           <- correct (flat)
    This function detects and unwraps those patterns.
    """
    args = tool_call.get("args", {})
    if not isinstance(args, dict):
        return tool_call

    # Pattern 1: {"tool_name": {...actual args...}}
    if tool_name in args and isinstance(args[tool_name], dict):
        fixed = dict(tool_call)
        fixed["args"] = args[tool_name]
        return fixed

    # Pattern 2: single-key wrapper under "args", "kwargs", "arguments"
    for wrap_key in ("args", "kwargs", "arguments"):
        if wrap_key in args and isinstance(args[wrap_key], dict) and len(args) == 1:
            fixed = dict(tool_call)
            fixed["args"] = args[wrap_key]
            return fixed

    # Pattern 3: single unknown key whose value is a dict of real params
    KNOWN_PARAMS = {
        "month", "start_date", "end_date", "category", "amount",
        "date", "id", "type", "subcategory", "note", "entries",
        "keyword", "limit", "offset", "top_n", "months", "period",
        "recurring", "tags", "payment_method", "order_by", "order_dir",
        "target_month", "lookback_months", "buffer_pct", "from_month",
        "to_month", "overwrite", "threshold",
    }
    if len(args) == 1:
        only_key, only_val = next(iter(args.items()))
        if isinstance(only_val, dict) and only_key not in KNOWN_PARAMS:
            fixed = dict(tool_call)
            fixed["args"] = only_val
            return fixed

    return tool_call


def _build_graph(tools: list):
    llm            = _build_llm()
    llm_with_tools = llm.bind_tools(tools)

    st.session_state._llm_ref = llm

    # Build name -> tool lookup for the custom dispatch node
    tool_map = {getattr(t, "name", str(t)): t for t in tools}

    async def chat_node(state: ChatState) -> dict:
        msgs = state["messages"]
        if not msgs or not isinstance(msgs[0], SystemMessage):
            msgs = [SystemMessage(content=SYSTEM_PROMPT)] + list(msgs)
        response = await llm_with_tools.ainvoke(msgs)
        return {"messages": [response]}

    async def tool_node(state: ChatState) -> dict:
        """
        Custom tool node that unwraps Gemini double-wrapped args
        before dispatching to each MCP tool.
        """
        import json
        last_msg = state["messages"][-1]
        results  = []

        for tc in getattr(last_msg, "tool_calls", []):
            name    = tc.get("name", "")
            call_id = tc.get("id", "")

            # Unwrap any malformed args from Gemini
            fixed = _fix_tool_call_args(tc, name)
            args  = fixed.get("args", {})

            tool = tool_map.get(name)
            if tool is None:
                content = f"Error: tool '{name}' not found."
            else:
                try:
                    raw     = await tool.ainvoke(args)
                    content = json.dumps(raw) if not isinstance(raw, str) else raw
                except Exception as exc:
                    content = f"Tool error: {exc}"

            results.append(
                ToolMessage(content=content, tool_call_id=call_id, name=name)
            )

        return {"messages": results}

    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tool_node", tool_node)
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges(
        "chat_node", tools_condition,
        {"tools": "tool_node", END: END},
    )
    graph.add_edge("tool_node", "chat_node")
    return graph.compile(checkpointer=InMemorySaver())


# ─────────────────────────────────────────────
# EVENT LOOP MANAGEMENT
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _get_event_loop():
    import threading
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    return loop


def _run(coro, timeout: int = 120):
    import concurrent.futures
    fut = asyncio.run_coroutine_threadsafe(coro, _get_event_loop())
    return fut.result(timeout=timeout)


# ─────────────────────────────────────────────
# MCP INIT
# ─────────────────────────────────────────────

import traceback as _tb


async def _init_mcp():
    client = MultiServerMCPClient(
        {
            "expense_tracker": {
                "transport": "stdio",
                "command":   sys.executable,
                "args":      [str(SERVER_PATH)],
            }
        }
    )
    tools = await client.get_tools()
    graph = _build_graph(tools)
    return client, tools, graph


def ensure_mcp() -> bool:
    if st.session_state.mcp_ready:
        return True
    if not SERVER_PATH.exists():
        st.session_state.error = f"server.py not found at: {SERVER_PATH}"
        return False
    try:
        client, tools, graph = _run(_init_mcp())
        st.session_state.mcp_client = client
        st.session_state.tools      = tools
        st.session_state.graph      = graph
        st.session_state.mcp_ready  = True
        return True
    except Exception as exc:
        st.session_state.error = "".join(
            _tb.format_exception(type(exc), exc, exc.__traceback__)
        )
        return False


# ─────────────────────────────────────────────
# AGENT INVOKE
# ─────────────────────────────────────────────

def invoke_agent(user_text: str) -> tuple[str, list[str]]:
    graph     = st.session_state.graph
    thread_id = st.session_state.thread_id
    cfg       = {"configurable": {"thread_id": thread_id}}

    async def _run_agent():
        return await graph.ainvoke(
            {"messages": [HumanMessage(content=user_text)]},
            config=cfg,
        )

    result   = _run(_run_agent())
    messages = result.get("messages", [])

    # Sync active tier
    llm_ref = st.session_state.get("_llm_ref")
    if llm_ref is not None:
        st.session_state.active_tier = llm_ref._tier

    # Collect tool calls
    tools_called: list[str] = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                tools_called.append(tc.get("name", "unknown"))

    # Pass 1: last AIMessage with text AND no pending tool_calls
    reply = ""
    for m in reversed(messages):
        if not isinstance(m, AIMessage):
            continue
        if getattr(m, "tool_calls", None):
            continue
        candidate = _extract_text(m.content)
        if candidate:
            reply = candidate
            break

    # Pass 2: fallback
    if not reply:
        for m in reversed(messages):
            if isinstance(m, AIMessage):
                candidate = _extract_text(m.content)
                if candidate:
                    reply = candidate
                    break

    if not reply:
        reply = "✅ Done."

    return reply, tools_called


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

def _tier_badge(tier: int) -> str:
    label = TIER_LABELS[tier]  if tier < len(TIER_LABELS)  else f"model-{tier}"
    cls   = TIER_COLOURS[tier] if tier < len(TIER_COLOURS) else "tool-badge"
    return f"<span class='tool-badge {cls}'>{label}</span>"


def render_sidebar():
    with st.sidebar:
        # ── Brand ────────────────────────────────
        st.markdown("""
        <div class="sidebar-brand">
            <div class="sidebar-brand-logo">Ft</div>
            <div>
                <div class="sidebar-brand-name">FinTrack</div>
                <div class="sidebar-brand-tagline">AI Personal Finance</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr class='subtle-divider'>", unsafe_allow_html=True)

        # ── Connection & LLM status ───────────────
        st.markdown("<div class='sidebar-section'>System</div>", unsafe_allow_html=True)
        if st.session_state.mcp_ready:
            tier = st.session_state.get("active_tier", 0)
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:8px'>"
                f"<span style='width:6px;height:6px;border-radius:50%;background:#34d399;"
                f"box-shadow:0 0 6px #34d39988;flex-shrink:0;display:inline-block'></span>"
                f"<span style='font-size:12px;color:#34d399;font-weight:500'>Connected</span>"
                f"<span style='font-size:11px;color:#2e3450'>· MCP running</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            # LLM chain
            chain_html = " <span style='color:#2e3450;font-size:10px'>→</span> ".join(
                f"<span class='tool-badge {TIER_COLOURS[i]}'>{TIER_LABELS[i]}</span>"
                for i in range(len(TIER_LABELS))
            )
            st.markdown(
                f"<div style='font-size:11px;color:#3a4060;font-family:IBM Plex Mono,monospace;"
                f"margin-bottom:4px'>LLM CHAIN</div>"
                f"<div style='margin-bottom:6px'>{chain_html}</div>"
                f"<div style='font-size:11px;color:#3a4060;font-family:IBM Plex Mono,monospace;"
                f"margin-bottom:4px'>ACTIVE</div>"
                f"<div style='margin-bottom:6px'>{_tier_badge(tier)}</div>"
                f"<div style='font-size:11px;color:#3a4060;font-family:IBM Plex Mono,monospace'>"
                f"{len(st.session_state.tools)} tools registered</div>",
                unsafe_allow_html=True,
            )
        elif st.session_state.error:
            st.markdown(
                "<span style='color:#f87171;font-weight:500;font-size:12px'>● Error</span>",
                unsafe_allow_html=True,
            )
            st.error(st.session_state.error)
        else:
            st.markdown(
                "<span style='color:#fbbf24;font-weight:500;font-size:12px'>● Connecting…</span>",
                unsafe_allow_html=True,
            )

        st.markdown("<hr class='subtle-divider'>", unsafe_allow_html=True)

        # ── Session stats ────────────────────────
        st.markdown("<div class='sidebar-section'>Session</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        col1.metric("Turns", st.session_state.turn_count)
        col2.metric("Tools", len(st.session_state.tool_log))

        st.markdown("<hr class='subtle-divider'>", unsafe_allow_html=True)

        # ── Quick actions ────────────────────────
        st.markdown("<div class='sidebar-section'>Quick actions</div>", unsafe_allow_html=True)
        for label, query in QUICK_ACTIONS:
            if st.button(label, key=f"qa_{label}"):
                st.session_state["_pending_query"] = query
                st.rerun()

        st.markdown("<hr class='subtle-divider'>", unsafe_allow_html=True)

        # ── Tool activity log ────────────────────
        st.markdown("<div class='sidebar-section'>Tool activity</div>", unsafe_allow_html=True)
        if st.session_state.tool_log:
            for turn_num, tools_used in reversed(st.session_state.tool_log[-10:]):
                badges = " ".join(
                    f"<span class='tool-badge'>{t}</span>" for t in tools_used
                )
                st.markdown(
                    f"<div style='font-size:10px;color:#2e3450;font-family:IBM Plex Mono,monospace;"
                    f"margin-bottom:3px;letter-spacing:0.06em'>TURN {turn_num}</div>{badges}"
                    f"<div style='height:6px'></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "<span style='font-size:11px;color:#2e3450;font-family:IBM Plex Mono,monospace'>"
                "No tool calls yet</span>",
                unsafe_allow_html=True,
            )

        st.markdown("<hr class='subtle-divider'>", unsafe_allow_html=True)

        # ── New conversation ─────────────────────
        if st.button("↺  Reset conversation", key="new_convo"):
            st.session_state.thread_id  = str(uuid.uuid4())
            st.session_state.messages   = []
            st.session_state.tool_log   = []
            st.session_state.turn_count = 0
            st.rerun()

        # ── Available tools expander ─────────────
        if st.session_state.tools:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            with st.expander("all tools", expanded=False):
                for t in st.session_state.tools:
                    name = getattr(t, "name", str(t))
                    desc = (getattr(t, "description", "") or "").split("\n")[0][:55]
                    st.markdown(
                        f"<div style='margin-bottom:4px'>"
                        f"<span class='tool-badge'>{name}</span>"
                        f"<span style='font-size:10px;color:#3a4060;display:block;"
                        f"margin-top:1px;padding-left:2px'>{desc}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )


# ─────────────────────────────────────────────
# CHAT HISTORY RENDERER
# ─────────────────────────────────────────────

def render_messages():
    for msg in st.session_state.messages:
        role    = msg["role"]
        content = msg["content"]
        tools   = msg.get("tools", [])

        with st.chat_message(role):
            st.markdown(content)
            if tools:
                badges = " ".join(
                    f"<span class='tool-badge'>{t}</span>" for t in tools
                )
                st.markdown(
                    f"<div class='tools-called-row'>"
                    f"<span class='tools-called-label'>called</span>{badges}"
                    f"</div>",
                    unsafe_allow_html=True,
                )


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

# ── Bootstrap MCP ───────────────────────────
if not st.session_state.mcp_ready:
    with st.spinner("Initialising FinTrack…"):
        ensure_mcp()
    if st.session_state.mcp_ready:
        st.rerun()

# ── Sidebar ─────────────────────────────────
render_sidebar()

# ── Main area header ────────────────────────
st.markdown("""
<div class='app-header'>
  <div class='fintrack-logo'>Ft</div>
  <div>
    <div class='app-title'>Fin<span>Track</span></div>
    <div class='app-subtitle'>AI · PERSONAL FINANCE · EXPENSE INTELLIGENCE</div>
  </div>
  <div style='margin-left:auto;display:flex;align-items:center;gap:8px'>
    <div class='status-dot'></div>
    <span style='font-size:11px;color:#2e3450;font-family:IBM Plex Mono,monospace'>LIVE</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Error state ─────────────────────────────
if st.session_state.error and not st.session_state.mcp_ready:
    st.error("**Could not start MCP server** — see error below:")
    st.code(st.session_state.error, language="text")
    st.info(
        f"**Checklist:**\n"
        f"- `server.py` at: `{SERVER_PATH}`\n"
        f"- MySQL running? Check `MYSQL_HOST` / `MYSQL_PASSWORD` in `.env`\n"
        f"- `fastmcp` installed? → `pip install fastmcp`\n"
        f"- `aiomysql` installed? → `pip install aiomysql`\n"
        f"- Set `MCP_SERVER_PATH` in `.env` if server.py is elsewhere\n"
        f"- Ollama running? → `ollama serve` (for local fallback)"
    )
    if st.button("Retry connection"):
        st.session_state.mcp_ready = False
        st.session_state.error     = None
        st.rerun()
    st.stop()

# ── Welcome message on first load ───────────
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown("""
<div class="welcome-card">
  <div class="welcome-title">Welcome to FinTrack</div>
  <div class="welcome-sub">
    Your AI-powered finance layer. Ask in plain language — I'll handle the rest.
  </div>
  <div class="capability-row">
    <span class="cap-pill">📝 Log expenses</span>
    <span class="cap-pill">📊 Spending analysis</span>
    <span class="cap-pill">🎯 Budget tracking</span>
    <span class="cap-pill">💡 Smart suggestions</span>
    <span class="cap-pill">📈 Trend reports</span>
  </div>
</div>

Try the **quick actions** in the sidebar, or just type naturally:

> *"I spent ₹450 on lunch today"*
> *"How's my budget looking this month?"*
> *"Show my top expenses for March"*
        """, unsafe_allow_html=True)

# ── Chat history ─────────────────────────────
render_messages()

# ── Handle pending query from quick-action ───
pending = st.session_state.pop("_pending_query", None)

# ── Chat input ───────────────────────────────
user_input = st.chat_input("Ask about your finances…") or pending

if user_input and st.session_state.mcp_ready:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        thinking = st.empty()
        thinking.markdown(
            "<div class='thinking-container'>"
            "<span class='thinking-label'>processing</span>"
            "<div class='thinking-dots'><span></span><span></span><span></span></div>"
            "</div>",
            unsafe_allow_html=True,
        )

        try:
            reply, tools_called = invoke_agent(user_input)
            thinking.empty()

            st.markdown(reply)

            if tools_called:
                badges = " ".join(
                    f"<span class='tool-badge'>{t}</span>" for t in tools_called
                )
                st.markdown(
                    f"<div class='tools-called-row'>"
                    f"<span class='tools-called-label'>called</span>{badges}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        except Exception as exc:
            thinking.empty()
            reply        = f"⚠️ Something went wrong: {exc}"
            tools_called = []
            st.error(reply)

    st.session_state.messages.append({
        "role":    "assistant",
        "content": reply,
        "tools":   tools_called,
    })
    st.session_state.turn_count += 1
    if tools_called:
        st.session_state.tool_log.append(
            (st.session_state.turn_count, tools_called)
        )

    st.rerun()
