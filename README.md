<div align="center">

<br/>

```
███████╗██╗███╗   ██╗████████╗██████╗  █████╗  ██████╗██╗  ██╗     █████╗ ██╗
██╔════╝██║████╗  ██║╚══██╔══╝██╔══██╗██╔══██╗██╔════╝██║ ██╔╝    ██╔══██╗██║
█████╗  ██║██╔██╗ ██║   ██║   ██████╔╝███████║██║     █████╔╝     ███████║██║
██╔══╝  ██║██║╚██╗██║   ██║   ██╔══██╗██╔══██║██║     ██╔═██╗     ██╔══██║██║
██║     ██║██║ ╚████║   ██║   ██║  ██║██║  ██║╚██████╗██║  ██╗    ██║  ██║██║
╚═╝     ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝   ╚═╝  ╚═╝╚═╝
```

### **Your money. Your conversation. Zero spreadsheets.**

<br/>

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-ReAct_Agent-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![MySQL](https://img.shields.io/badge/MySQL-4479A1?style=for-the-badge&logo=mysql&logoColor=white)](https://mysql.com)
[![Gemini](https://img.shields.io/badge/Gemini_2.0_Flash-Primary_LLM-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev)
[![Groq](https://img.shields.io/badge/Groq_Llama_3.3-Fallback_LLM-F55036?style=for-the-badge)](https://groq.com)
[![MCP](https://img.shields.io/badge/Model_Context_Protocol-Tool_Bridge-6B21A8?style=for-the-badge)](https://modelcontextprotocol.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

<br/>

> *"Spent ₹4,200 on food this month — split it: ₹800 Zomato, ₹1,100 groceries, ₹2,300 eating out"*
> — and it's logged. No forms. No dropdowns. Just that.

<br/>

</div>

---

## ✦ What is FinTrack AI?

FinTrack AI is a **fully local, AI-powered personal finance system** that runs entirely through natural language. You describe your money life in plain sentences — it logs, analyses, budgets, and advises. No forms. No spreadsheets. No friction.

Under the hood, a **LangGraph ReAct agent** powered by **Gemini 2.0 Flash** (with a seamless Groq fallback) connects to **30 typed MCP tools** that read and write to a **MySQL database** — all bridged through the **Model Context Protocol**. The result: a conversation that actually *does* things.

<br/>

---

## ✦ Feature Overview

<br/>

### 💬 Natural Language Everything
No UI forms. You talk, it acts.

```
"Add ₹450 Dunzo groceries, ₹1200 electric bill, and ₹300 coffee this week"
→ Agent calls get_categories → add_expenses_bulk → confirms all 3 in one shot
```

```
"What did I spend on food last month?"
→ summarize_by_category(category="Food", month=last_month)
→ Full breakdown returned in natural language
```

<br/>

### 🧾 Transaction Management
- Log **expenses, income, and savings** in a single sentence
- **Bulk ingestion** — mention 10 items, one database round-trip (`add_expenses_bulk`)
- Auto-fetches existing categories before every log to prevent spelling drift
- Tags, notes, subcategories — all accepted inline in natural language
- Full-text search across notes, tags, categories (`search_expenses`)

<br/>

### 📊 Smart Analytics Engine

| Tool | What It Does |
|---|---|
| `summarize_by_category` | Spending breakdown by category for any date range |
| `get_cashflow` | Income / Expenses / Savings / Net position / Savings rate % |
| `get_top_expenses` | Your biggest individual transactions |
| `summarize_by_period` | Bucket spending by day, week, or month |
| `get_monthly_trend` | Track a single category's spending over time |

<br/>

### 🎯 Live Budget System

```
You:   "Am I on track with my budget?"
Agent: → get_budget_status()

Category       Budget    Spent    Pace         Status
─────────────────────────────────────────────────────
Food           ₹8,000    ₹5,200   ON PACE      ✅ HEALTHY
Entertainment  ₹3,000    ₹2,900   OVERSPENDING ⚠️  CRITICAL
Transport      ₹2,000    ₹800     UNDERSPENDING ✅ ON TRACK
```

Pace labels are calculated relative to **how many days have elapsed** in the current month — not just raw totals.

<br/>

### 🤖 AI Budget Suggestions

```
"Suggest budgets based on my last 3 months"
→ suggest_budgets_from_history(months=3, buffer_pct=15)
→ Per-category averages + 15% buffer shown for your review
→ "Apply them" → set_budgets_bulk() in one call
```

<br/>

### 🏦 Savings Tracking

- Log savings events as moments: *"Cooked at home instead of ordering, saved ₹200"*
- Set monthly savings goals with `set_savings_goal`
- `get_savings_summary` — actual vs goal, pace-tracked identically to the budget system

<br/>

### 🔄 Gemini → Groq Auto-Fallback

```python
class FallbackLLM:
    # Primary: Gemini 2.0 Flash
    # On 429/quota error: silently switches to Groq Llama 3.3 70B
    # Zero interruption. Zero manual intervention.
```

Your conversation **never breaks** due to API rate limits.

<br/>

### 🧠 Persistent Conversation Memory

- `InMemorySaver` keeps full message history per thread ID
- Ask follow-up questions — the agent remembers everything from earlier in the session
- **"New Conversation"** button in sidebar generates a fresh thread for a clean slate

<br/>

---

## ✦ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        STREAMLIT UI (app.py)                    │
│   Chat Interface · Sidebar · Quick Actions · Tool Activity Log  │
└───────────────────────────┬─────────────────────────────────────┘
                            │  sync ↔ async bridge
                            │  (daemon thread + event loop)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LANGGRAPH ReAct AGENT                        │
│                                                                 │
│   ┌─────────────┐    tool_calls?    ┌──────────────────────┐   │
│   │  chat_node  │ ──────────────→  │     tool_node        │   │
│   │  (LLM)      │ ←────────────── │  (executes MCP tools) │   │
│   └─────────────┘    tool_results  └──────────────────────┘   │
│                                                                 │
│   Memory: InMemorySaver (per thread_id)                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │  langchain-mcp-adapters (stdio)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  FastMCP SERVER (server.py)                     │
│                                                                 │
│   expenses.py (14 tools)     budgets.py (16 tools)             │
│   ┌────────────────────┐     ┌────────────────────────────┐   │
│   │ add_expense        │     │ set_budget                 │   │
│   │ add_expenses_bulk  │     │ get_budget_status          │   │
│   │ get_expenses       │     │ get_budget_vs_actual       │   │
│   │ search_expenses    │     │ suggest_budgets_from_hist..│   │
│   │ summarize_by_cat   │     │ set_budgets_bulk           │   │
│   │ get_cashflow       │     │ add_saving                 │   │
│   │ get_top_expenses   │     │ set_savings_goal           │   │
│   │ get_monthly_trend  │     │ get_savings_summary        │   │
│   │ ...                │     │ ...                        │   │
│   └─────────┬──────────┘     └────────────┬───────────────┘   │
└─────────────┼──────────────────────────────┼────────────────────┘
              │                              │
              └──────────────┬───────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MySQL + aiomysql (db.py)                     │
│                                                                 │
│   expenses table                 budgets table                  │
│   ─────────────────────          ─────────────────────         │
│   id, amount, category,          id, category, monthly_limit,  │
│   subcategory, note, tags,       savings_goal, created_at      │
│   type, date, created_at                                        │
│                                                                 │
│   FULLTEXT index on (note, tags)                               │
│   AUTO-CREATED on startup                                       │
└─────────────────────────────────────────────────────────────────┘
```

<br/>

---

## ✦ Tech Stack

| Layer | Technology | Role |
|---|---|---|
| **UI** | Streamlit | Chat interface, sidebar, session state, quick actions |
| **Agent** | LangGraph (ReAct) | Reason+Act loop, tool orchestration |
| **Primary LLM** | Gemini 2.0 Flash | Main reasoning + tool calling |
| **Fallback LLM** | Groq · Llama 3.3 70B | Automatic quota-error fallback |
| **Tool Protocol** | MCP (Model Context Protocol) | Agent ↔ tool server bridge |
| **Tool Server** | FastMCP | 30 typed async tools with auto-schema |
| **Database** | MySQL + aiomysql | Persistent storage, async connection pool |
| **Memory** | LangGraph InMemorySaver | Per-session conversation history |

<br/>

---

## ✦ Project Structure

```
mcp-expense-tracker/
│
├── app.py              ← Streamlit UI + LangGraph wiring
│                         (async bridge, session state, tool activity log)
│
├── client.py           ← CLI version of the same agent
│                         (terminal chat, identical agent under the hood)
│
├── server.py           ← FastMCP server entry point
│                         (spawned as subprocess, communicates via stdio)
│
├── db.py               ← MySQL async pool
│                         (lazy init, auto-creates tables on startup)
│
├── expenses.py         ← 14 MCP expense tools
│                         (CRUD, analytics, search, cashflow)
│
├── budgets.py          ← 16 MCP budget + savings tools
│                         (budget CRUD, status, AI suggestions, savings)
│
├── requirements.txt    ← All dependencies pinned
│
└── .env                ← API keys + DB credentials (never commit this)
```

<br/>

---

## ✦ Getting Started

### Prerequisites

- Python 3.11+
- MySQL server running locally (or remote)
- Gemini API key ([get one free](https://ai.google.dev))
- Groq API key ([get one free](https://console.groq.com)) — for fallback

<br/>

### 1. Clone & Install

```bash
git clone https://github.com/Satyam-Singh-x/FinTrack-AI.git
cd mcp-expense-tracker
pip install -r requirements.txt
```

<br/>

### 2. Configure Environment

Create a `.env` file in the project root:

```env
# LLM Keys
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# MySQL
DB_HOST=localhost
DB_PORT=3306
DB_USER=your_mysql_user
DB_PASSWORD=your_mysql_password
DB_NAME=fintrack
```

> Tables are **auto-created** on first run. No migrations needed.

<br/>

### 3. Run

**Web UI (Streamlit):**
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

**CLI (Terminal chat):**
```bash
python client.py
```

<br/>

---

## ✦ Example Conversations

```
You:    "I spent ₹250 on coffee, ₹1800 on groceries, and paid ₹4500 rent advance today"

Agent:  Adding 3 transactions...
        ✓ Coffee (Food & Drink) — ₹250
        ✓ Groceries (Food & Drink) — ₹1,800
        ✓ Rent Advance (Housing) — ₹4,500
        All logged! Your Food & Drink total this month is now ₹6,240.
```

```
You:    "How's my spending looking vs last month?"

Agent:  Here's your month-over-month comparison:
        📈 Food went up 23% (₹5,070 → ₹6,240)
        📉 Transport dropped 40% (₹2,100 → ₹1,260)
        ➡️  Entertainment stayed flat (₹1,800 → ₹1,750)
        Net spend this month: ₹28,400 vs ₹26,100 last month (+8.8%)
```

```
You:    "Find all my Swiggy orders"

Agent:  Found 7 transactions tagged or noted with Swiggy:
        • Aug 12 — Swiggy dinner — ₹480
        • Aug 15 — Swiggy lunch (office) — ₹320
        • Aug 19 — Swiggy late night — ₹650
        ... (4 more)
        Total: ₹3,210 across 7 orders
```

```
You:    "Suggest budgets for next month based on my history"

Agent:  Based on your last 3 months + 15% buffer:
        Food & Drink    avg ₹5,400  →  suggested ₹6,210
        Transport       avg ₹1,800  →  suggested ₹2,070
        Entertainment   avg ₹1,750  →  suggested ₹2,012
        Housing         avg ₹12,000 →  suggested ₹12,000 (fixed, no buffer)

        Apply all of these?

You:    "Yes, apply them"

Agent:  ✓ 4 budgets set for September.
```

<br/>

---

## ✦ The 30 MCP Tools

<details>
<summary><strong>💸 Expense Tools (14)</strong> — click to expand</summary>

| Tool | Description |
|---|---|
| `add_expense` | Log a single transaction (expense, income, or saving) |
| `add_expenses_bulk` | Log multiple transactions in one database call |
| `get_expenses` | Retrieve transactions with filters (date, category, type, limit) |
| `get_categories` | List all existing categories (called before every log) |
| `search_expenses` | Full-text search on notes + tags; LIKE fallback on category |
| `update_expense` | Edit an existing transaction by ID |
| `delete_expense` | Remove a transaction by ID |
| `summarize_by_category` | Spending totals grouped by category for a date range |
| `summarize_by_period` | Bucket spending by day / week / month |
| `get_cashflow` | Income, expenses, savings, net, savings rate % |
| `get_top_expenses` | Top N largest individual transactions |
| `get_monthly_trend` | Single category spending trend over time |
| `get_spending_by_tag` | Aggregate spend grouped by tag |
| `get_recent_activity` | Last N transactions as a quick recap |

</details>

<details>
<summary><strong>🎯 Budget & Savings Tools (16)</strong> — click to expand</summary>

| Tool | Description |
|---|---|
| `set_budget` | Set monthly spending limit for a category |
| `set_budgets_bulk` | Set multiple budgets in one call |
| `get_budgets` | List all configured budgets |
| `update_budget` | Modify an existing budget |
| `delete_budget` | Remove a budget |
| `get_budget_status` | Live actual vs budget with pace + status labels |
| `get_budget_vs_actual` | Full review including unbudgeted spend categories |
| `suggest_budgets_from_history` | AI-computed suggestions from past N months |
| `add_saving` | Log a savings event with description and amount |
| `get_savings` | Retrieve savings logs with filters |
| `set_savings_goal` | Set a monthly savings target |
| `get_savings_goal` | Retrieve current savings goal |
| `get_savings_summary` | Actual vs goal with pace tracking |
| `update_saving` | Edit an existing savings log |
| `delete_saving` | Remove a savings log |
| `get_savings_trend` | Monthly savings trend over time |

</details>

<br/>

---

## ✦ How the ReAct Loop Works

```
User message
     │
     ▼
┌──────────────┐
│  chat_node   │  LLM reads message + full history + available tool schemas
│  (Gemini /   │  → Decides: respond directly OR call one/more tools
│   Groq)      │
└──────┬───────┘
       │ tool_calls in response?
       │
   YES ▼                          NO ▼
┌──────────────┐            Final response
│  tool_node   │            sent to user
│  (MCP exec)  │
└──────┬───────┘
       │ tool results appended to history
       │
       └──────────────────────────────→ back to chat_node
                                        (loop until no more tool calls)
```

The agent can chain multiple tool calls in a single turn. For example, asking *"set budgets based on my history and tell me my cashflow"* might trigger: `suggest_budgets_from_history` → `set_budgets_bulk` → `get_cashflow` — all in one response.

<br/>

---

## ✦ Why MCP?

The agent **never touches the database directly**. It only knows about tools — their names, parameters, and return shapes. This means:

- **Swap the database** (Postgres, SQLite) without touching agent code
- **Add new tools** by decorating a Python function with `@mcp.tool` — the agent discovers them automatically
- **Run server.py standalone** for testing or connecting other MCP-compatible clients
- **Schema validation** is handled by FastMCP automatically — malformed tool calls never reach the DB

<br/>

---

## ✦ Requirements

```txt
streamlit
langchain-core
langchain-google-genai
langchain-groq
langgraph
langchain-mcp-adapters
fastmcp
aiomysql
python-dotenv
```

Install everything:
```bash
pip install -r requirements.txt
```

<br/>

---

## ✦ Roadmap

- [ ] Recurring transaction detection
- [ ] CSV / bank statement import
- [ ] Multi-currency support
- [ ] Streamlit Cloud / Docker deployment guide
- [ ] Per-user multi-tenancy
- [ ] WhatsApp / Telegram bot interface (MCP tools are already decoupled — any client works)
- [ ] Monthly PDF reports

<br/>

---

## ✦ Contributing

PRs welcome. Please open an issue first for major changes.

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

<br/>

---

## ✦ License

MIT — see [LICENSE](LICENSE) for details.

<br/>

---

<div align="center">

**Built with LangGraph · FastMCP · Gemini · Groq · Streamlit · MySQL**

*Stop tracking money. Start talking about it.*

</div>
