"""
budgets.py
──────────
All budget and savings-goal MCP tools for the Expense Tracker server.
Registered onto the FastMCP instance passed in from server.py via register(mcp).

Tool index
──────────
BUDGET CRUD
  1.  set_budget                  — upsert a monthly spending limit for a category
  2.  set_budgets_bulk            — upsert many budgets in one call
  3.  get_budget_by_id            — fetch one budget/goal row by ID
  4.  list_budgets                — list all budgets for a month
  5.  update_budget               — change the amount of an existing budget
  6.  delete_budget               — delete by ID or month+category

BUDGET ANALYTICS
  7.  get_budget_status           — live actual vs budget with pace & status labels
  8.  get_budget_vs_actual        — full side-by-side month review incl. unbudgeted
  9.  get_overbudget_categories   — only categories that breached a threshold %
  10. get_budget_trend            — month-over-month budget vs actual for a category

SMART BUDGET TOOLS
  11. suggest_budgets_from_history — suggest limits based on past avg spending
  12. copy_budgets_from_month      — carry forward all budgets to a new month

SAVINGS
  13. add_saving                  — log a saving event with date, amount, category
  14. set_savings_goal            — set a monthly savings target per category
  15. get_savings_summary         — actual savings vs goal this month
  16. get_savings_trend           — month-over-month savings history
"""

from __future__ import annotations

import calendar
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import aiomysql

from db import get_pool


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _clean(rows: list[dict] | dict) -> Any:
    """Convert Decimal → float and date/datetime → ISO string recursively."""
    if isinstance(rows, list):
        return [_clean(r) for r in rows]
    if isinstance(rows, dict):
        return {
            k: (
                float(v)       if isinstance(v, Decimal)          else
                v.isoformat()  if isinstance(v, (date, datetime)) else
                v
            )
            for k, v in rows.items()
        }
    return rows


def _validate_date(d: str, field: str = "date") -> None:
    try:
        datetime.strptime(d, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"'{field}' must be YYYY-MM-DD, got: {d!r}")


def _validate_amount(a: float, field: str = "amount") -> None:
    if a < 0:
        raise ValueError(f"'{field}' must be >= 0, got: {a}")


def _month_start(month_str: str) -> str:
    """
    Normalise any of 'YYYY-MM' or 'YYYY-MM-DD' to 'YYYY-MM-01'.
    This keeps all month keys consistent across the budgets table.
    """
    s = month_str.strip()
    try:
        if len(s) == 7:
            datetime.strptime(s, "%Y-%m")
            return f"{s}-01"
        elif len(s) == 10:
            datetime.strptime(s, "%Y-%m-%d")
            return f"{s[:7]}-01"
    except ValueError:
        pass
    raise ValueError(f"'month' must be YYYY-MM or YYYY-MM-DD, got: {month_str!r}")


def _month_end(month_start: str) -> str:
    """Return the first day of the NEXT month — used for exclusive range queries."""
    y, m = int(month_start[:4]), int(month_start[5:7])
    return f"{y+1}-01-01" if m == 12 else f"{y}-{m+1:02d}-01"


def _days_stats(month_start: str) -> dict:
    """Return days_in_month, days_elapsed, first and last date of the month."""
    y, m    = int(month_start[:4]), int(month_start[5:7])
    dim     = calendar.monthrange(y, m)[1]
    first   = date(y, m, 1)
    last    = date(y, m, dim)
    today   = date.today()
    elapsed = (min(today, last) - first).days + 1
    return {"days_in_month": dim, "days_elapsed": elapsed, "first": first, "last": last}


def _status_label(pct: float) -> str:
    if pct >= 100: return "OVER BUDGET"
    if pct >= 85:  return "CRITICAL"
    if pct >= 60:  return "ON TRACK"
    return              "HEALTHY"


def _pace_label(actual: float, expected: float) -> str:
    if actual > expected * 1.05: return "OVERSPENDING"
    if actual < expected * 0.85: return "UNDERSPENDING"
    return                              "ON PACE"


def _savings_status(pct: float) -> str:
    if pct >= 100: return "GOAL MET"
    if pct >= 75:  return "ON TRACK"
    if pct >= 40:  return "PROGRESSING"
    return              "BEHIND"


# ─────────────────────────────────────────────
# REGISTRATION ENTRY POINT
# ─────────────────────────────────────────────

def register(mcp) -> None:
    """Mount all budget and savings tools onto the FastMCP instance."""

    # ════════════════════════════════════════════════════════════
    # BUDGET CRUD
    # ════════════════════════════════════════════════════════════

    # ── 1. SET BUDGET ────────────────────────────────────────────
    @mcp.tool()
    async def set_budget(
        month:       str,
        category:    str,
        amount:      float,
        subcategory: str = "",
        note:        str = "",
    ) -> dict:
        """
        Create or update a monthly spending limit for a category.
        Safe to call repeatedly — updates in place if already exists (upsert).

        Parameters
        ──────────
        month       : 'YYYY-MM' or 'YYYY-MM-DD'   e.g. '2025-03'
        category    : e.g. 'Food', 'Transport'
        amount      : spending limit in your currency  (>= 0)
        subcategory : optional finer granularity  e.g. 'Groceries'
        note        : optional note about this budget
        """
        month = _month_start(month)
        _validate_amount(amount)
        if not category.strip():
            raise ValueError("'category' cannot be empty.")

        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO budgets (month, type, category, subcategory, amount, note)
                    VALUES (%s, 'budget', %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        amount = VALUES(amount),
                        note   = VALUES(note)
                    """,
                    (month, category.strip(), subcategory.strip(), amount, note.strip()),
                )
                action = "updated" if cur.rowcount == 2 else "created"
                return {
                    "status":      "ok",
                    "action":      action,
                    "month":       month,
                    "category":    category.strip(),
                    "subcategory": subcategory.strip(),
                    "amount":      amount,
                }


    # ── 2. SET BUDGETS BULK ──────────────────────────────────────
    @mcp.tool()
    async def set_budgets_bulk(entries: list[dict]) -> dict:
        """
        Upsert multiple monthly spending budgets in a single call.

        Each entry follows the same schema as set_budget:
            [
              {"month":"2025-03","category":"Food","amount":5000},
              {"month":"2025-03","category":"Transport","subcategory":"Uber","amount":1500},
              {"month":"2025-03","category":"Entertainment","amount":2000,"note":"movies+OTT"}
            ]
        """
        if not entries:
            raise ValueError("'entries' list is empty.")

        rows = []
        for i, e in enumerate(entries):
            month  = _month_start(e.get("month", ""))
            cat    = e.get("category", "").strip()
            subcat = e.get("subcategory", "").strip()
            amt    = float(e.get("amount", -1))
            note   = e.get("note", "").strip()

            if not cat:
                raise ValueError(f"entries[{i}].category cannot be empty.")
            _validate_amount(amt, f"entries[{i}].amount")
            rows.append((month, cat, subcat, amt, note))

        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.executemany(
                    """
                    INSERT INTO budgets (month, type, category, subcategory, amount, note)
                    VALUES (%s, 'budget', %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        amount = VALUES(amount),
                        note   = VALUES(note)
                    """,
                    rows,
                )
                return {"status": "ok", "upserted": len(rows)}


    # ── 3. GET BUDGET BY ID ──────────────────────────────────────
    @mcp.tool()
    async def get_budget_by_id(id: int) -> dict:
        """Fetch a single budget or savings-goal row by its primary-key ID."""
        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute("SELECT * FROM budgets WHERE id = %s", (id,))
                row = await cur.fetchone()
        if row is None:
            raise ValueError(f"No budget found with id={id}")
        return _clean(row)


    # ── 4. LIST BUDGETS ──────────────────────────────────────────
    @mcp.tool()
    async def list_budgets(
        month:       str,
        type:        str        = "budget",
        category:    str | None = None,
        subcategory: str | None = None,
    ) -> list[dict]:
        """
        List all budgets or savings goals set for a given month.

        Parameters
        ──────────
        month       : 'YYYY-MM' or 'YYYY-MM-DD'
        type        : 'budget' (default) | 'savings_goal'
        category    : optional exact-match filter
        subcategory : optional exact-match filter
        """
        month = _month_start(month)
        if type not in ("budget", "savings_goal"):
            raise ValueError("'type' must be 'budget' or 'savings_goal'.")

        clauses = ["month = %s", "type = %s"]
        params  = [month, type]
        if category:
            clauses.append("category = %s"); params.append(category)
        if subcategory:
            clauses.append("subcategory = %s"); params.append(subcategory)

        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(
                    f"""
                    SELECT id, month, type, category, subcategory, amount, note, created_at
                    FROM budgets
                    WHERE {" AND ".join(clauses)}
                    ORDER BY category ASC, subcategory ASC
                    """,
                    params,
                )
                rows = await cur.fetchall()
        return _clean(list(rows))


    # ── 5. UPDATE BUDGET ─────────────────────────────────────────
    @mcp.tool()
    async def update_budget(
        id:     int,
        amount: float,
        note:   str = "",
    ) -> dict:
        """
        Update the amount (and optionally the note) of an existing
        budget or savings-goal entry by its ID.

        Use list_budgets() to find the ID first.
        """
        _validate_amount(amount)
        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE budgets SET amount = %s, note = %s WHERE id = %s",
                    (amount, note.strip(), id),
                )
                if cur.rowcount == 0:
                    raise ValueError(f"No budget found with id={id}")
        return {"status": "ok", "updated_id": id, "new_amount": amount}


    # ── 6. DELETE BUDGET ─────────────────────────────────────────
    @mcp.tool()
    async def delete_budget(
        id:          int | None = None,
        month:       str | None = None,
        category:    str | None = None,
        type:        str        = "budget",
        subcategory: str | None = None,
    ) -> dict:
        """
        Delete a budget or savings-goal entry.

        Option A — by ID (precise):
            delete_budget(id=7)

        Option B — by month + category (flexible):
            delete_budget(month='2025-03', category='Food')
            delete_budget(month='2025-03', category='Food', type='savings_goal')

        At least one of (id) or (month + category) must be provided.
        """
        if id is None and not (month and category):
            raise ValueError("Provide 'id'  OR  both 'month' and 'category'.")
        if type not in ("budget", "savings_goal"):
            raise ValueError("'type' must be 'budget' or 'savings_goal'.")

        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                if id is not None:
                    await cur.execute("DELETE FROM budgets WHERE id = %s", (id,))
                    if cur.rowcount == 0:
                        raise ValueError(f"No budget found with id={id}")
                    return {"status": "ok", "deleted_id": id}
                else:
                    ms = _month_start(month)
                    clauses = ["month = %s", "type = %s", "category = %s"]
                    params  = [ms, type, category.strip()]
                    if subcategory:
                        clauses.append("subcategory = %s")
                        params.append(subcategory.strip())
                    await cur.execute(
                        f"DELETE FROM budgets WHERE {' AND '.join(clauses)}", params
                    )
                    return {
                        "status":       "ok",
                        "deleted_rows": cur.rowcount,
                        "month":        ms,
                        "category":     category.strip(),
                    }


    # ════════════════════════════════════════════════════════════
    # BUDGET ANALYTICS
    # ════════════════════════════════════════════════════════════

    # ── 7. GET BUDGET STATUS ─────────────────────────────────────
    @mcp.tool()
    async def get_budget_status(
        month:       str,
        category:    str | None = None,
        subcategory: str | None = None,
    ) -> list[dict]:
        """
        Live actual spending vs budget for every category in a month.

        Returns per category:
          budget_amount   — the limit set
          actual_spent    — what was actually spent so far
          remaining       — budget minus actual
          pct_used        — percentage of budget consumed
          status          — HEALTHY / ON TRACK / CRITICAL / OVER BUDGET
          expected_spend  — where you should be based on days elapsed in month
          pace_status     — OVERSPENDING / ON PACE / UNDERSPENDING

        This is the go-to tool for mid-month budget check-ins.
        """
        ms = _month_start(month)
        me = _month_end(ms)
        ds = _days_stats(ms)

        cat_filter, params_extra = "", []
        if category:
            cat_filter += " AND b.category = %s";    params_extra.append(category)
        if subcategory:
            cat_filter += " AND b.subcategory = %s"; params_extra.append(subcategory)

        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(
                    f"""
                    SELECT
                        b.id,
                        b.category,
                        b.subcategory,
                        b.amount                    AS budget_amount,
                        b.note                      AS budget_note,
                        COALESCE(SUM(e.amount), 0)  AS actual_spent
                    FROM budgets b
                    LEFT JOIN expenses e
                        ON  e.category    = b.category
                        AND (b.subcategory = '' OR e.subcategory = b.subcategory)
                        AND e.type        = 'expense'
                        AND e.date       >= %s
                        AND e.date        < %s
                    WHERE b.month = %s
                      AND b.type  = 'budget'
                    {cat_filter}
                    GROUP BY b.id, b.category, b.subcategory, b.amount, b.note
                    ORDER BY b.category ASC, b.subcategory ASC
                    """,
                    [ms, me, ms] + params_extra,
                )
                rows = await cur.fetchall()

        result = []
        for r in rows:
            budget    = float(r["budget_amount"])
            actual    = float(r["actual_spent"])
            remaining = round(budget - actual, 2)
            pct       = round(actual / budget * 100, 2) if budget else 0.0
            expected  = round(budget * ds["days_elapsed"] / ds["days_in_month"], 2)
            result.append({
                "id":             r["id"],
                "category":       r["category"],
                "subcategory":    r["subcategory"],
                "budget_note":    r["budget_note"],
                "budget_amount":  budget,
                "actual_spent":   round(actual, 2),
                "remaining":      remaining,
                "pct_used":       f"{pct}%",
                "status":         _status_label(pct),
                "days_in_month":  ds["days_in_month"],
                "days_elapsed":   ds["days_elapsed"],
                "expected_spend": expected,
                "pace_status":    _pace_label(actual, expected),
            })
        return result


    # ── 8. GET BUDGET VS ACTUAL ──────────────────────────────────
    @mcp.tool()
    async def get_budget_vs_actual(month: str) -> dict:
        """
        Full side-by-side budget vs actual for every category in a month.
        Also surfaces categories where money was spent but NO budget was set.
        Ideal for end-of-month review conversations.
        """
        ms = _month_start(month)
        me = _month_end(ms)

        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:

                # Categories WITH a budget
                await cur.execute(
                    """
                    SELECT
                        b.category, b.subcategory,
                        b.amount                    AS budget_amount,
                        COALESCE(SUM(e.amount), 0)  AS actual_spent
                    FROM budgets b
                    LEFT JOIN expenses e
                        ON  e.category = b.category
                        AND e.type     = 'expense'
                        AND e.date    >= %s AND e.date < %s
                    WHERE b.month = %s AND b.type = 'budget'
                    GROUP BY b.category, b.subcategory, b.amount
                    ORDER BY b.category ASC
                    """,
                    (ms, me, ms),
                )
                budgeted = list(await cur.fetchall())

                # Categories with spending but NO budget set
                await cur.execute(
                    """
                    SELECT
                        e.category, e.subcategory,
                        0               AS budget_amount,
                        SUM(e.amount)   AS actual_spent
                    FROM expenses e
                    LEFT JOIN budgets b
                        ON  b.category = e.category
                        AND b.month    = %s
                        AND b.type     = 'budget'
                    WHERE e.type  = 'expense'
                      AND e.date >= %s AND e.date < %s
                      AND b.id IS NULL
                    GROUP BY e.category, e.subcategory
                    ORDER BY actual_spent DESC
                    """,
                    (ms, ms, me),
                )
                unbudgeted = list(await cur.fetchall())

        all_rows     = budgeted + unbudgeted
        total_budget = sum(float(r["budget_amount"]) for r in all_rows)
        total_actual = sum(float(r["actual_spent"])  for r in all_rows)

        breakdown = []
        for r in all_rows:
            budget = float(r["budget_amount"])
            actual = float(r["actual_spent"])
            pct    = round(actual / budget * 100, 2) if budget else None
            breakdown.append({
                "category":      r["category"],
                "subcategory":   r["subcategory"],
                "budget_amount": budget,
                "actual_spent":  round(actual, 2),
                "difference":    round(budget - actual, 2),
                "pct_used":      f"{pct}%" if pct is not None else "No budget set",
                "status":        _status_label(pct) if pct is not None else "UNBUDGETED",
            })

        return {
            "month":        ms,
            "total_budget": round(total_budget, 2),
            "total_actual": round(total_actual, 2),
            "total_saved":  round(total_budget - total_actual, 2),
            "overall_pct":  f"{round(total_actual / total_budget * 100, 2)}%" if total_budget else "N/A",
            "breakdown":    breakdown,
        }


    # ── 9. GET OVERBUDGET CATEGORIES ────────────────────────────
    @mcp.tool()
    async def get_overbudget_categories(
        month:     str,
        threshold: float = 100.0,
    ) -> dict:
        """
        Return only categories that have crossed a usage threshold.

        Parameters
        ──────────
        month     : 'YYYY-MM' or 'YYYY-MM-DD'
        threshold : % above which a category is flagged  (default 100 = actually over)
                    Pass 85 to catch categories approaching their limit early.
        """
        if not (0 <= threshold <= 500):
            raise ValueError("'threshold' must be between 0 and 500.")

        ms = _month_start(month)
        me = _month_end(ms)
        ds = _days_stats(ms)

        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(
                    """
                    SELECT
                        b.category, b.subcategory,
                        b.amount                    AS budget_amount,
                        COALESCE(SUM(e.amount), 0)  AS actual_spent
                    FROM budgets b
                    LEFT JOIN expenses e
                        ON  e.category = b.category
                        AND e.type     = 'expense'
                        AND e.date    >= %s AND e.date < %s
                    WHERE b.month = %s AND b.type = 'budget'
                    GROUP BY b.category, b.subcategory, b.amount
                    HAVING actual_spent >= (b.amount * %s / 100)
                    ORDER BY (actual_spent / b.amount) DESC
                    """,
                    (ms, me, ms, threshold),
                )
                rows = await cur.fetchall()

        flagged = []
        for r in rows:
            budget   = float(r["budget_amount"])
            actual   = float(r["actual_spent"])
            pct      = round(actual / budget * 100, 2) if budget else 0.0
            expected = round(budget * ds["days_elapsed"] / ds["days_in_month"], 2)
            flagged.append({
                "category":      r["category"],
                "subcategory":   r["subcategory"],
                "budget_amount": budget,
                "actual_spent":  round(actual, 2),
                "overspent_by":  round(actual - budget, 2),
                "pct_used":      f"{pct}%",
                "status":        _status_label(pct),
                "pace_status":   _pace_label(actual, expected),
            })

        return {
            "month":      ms,
            "threshold":  f"{threshold}%",
            "flagged":    len(flagged),
            "categories": flagged,
        }


    # ── 10. GET BUDGET TREND ─────────────────────────────────────
    @mcp.tool()
    async def get_budget_trend(
        category: str,
        months:   int = 6,
    ) -> list[dict]:
        """
        Month-over-month budget vs actual trend for a specific category.
        Useful for spotting whether spending in a category is growing or shrinking.

        Parameters
        ──────────
        category : category to track
        months   : how many past months to include  (default 6, max 24)
        """
        if not category.strip():
            raise ValueError("'category' cannot be empty.")
        months = min(max(1, months), 24)

        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(
                    """
                    SELECT
                        b.month,
                        b.amount                    AS budget_amount,
                        COALESCE(SUM(e.amount), 0)  AS actual_spent
                    FROM budgets b
                    LEFT JOIN expenses e
                        ON  e.category = b.category
                        AND e.type     = 'expense'
                        AND DATE_FORMAT(e.date, '%%Y-%%m-01') = b.month
                    WHERE b.category = %s
                      AND b.type     = 'budget'
                      AND b.month   >= DATE_FORMAT(
                            DATE_SUB(CURDATE(), INTERVAL %s MONTH), '%%Y-%%m-01'
                          )
                    GROUP BY b.month, b.amount
                    ORDER BY b.month ASC
                    """,
                    (category.strip(), months),
                )
                rows = await cur.fetchall()

        result = []
        for r in rows:
            budget = float(r["budget_amount"])
            actual = float(r["actual_spent"])
            pct    = round(actual / budget * 100, 2) if budget else 0.0
            result.append({
                "month":         r["month"].isoformat() if hasattr(r["month"], "isoformat") else str(r["month"]),
                "budget_amount": budget,
                "actual_spent":  round(actual, 2),
                "difference":    round(budget - actual, 2),
                "pct_used":      f"{pct}%",
                "status":        _status_label(pct),
            })
        return result


    # ════════════════════════════════════════════════════════════
    # SMART BUDGET TOOLS
    # ════════════════════════════════════════════════════════════

    # ── 11. SUGGEST BUDGETS FROM HISTORY ────────────────────────
    @mcp.tool()
    async def suggest_budgets_from_history(
        target_month:    str,
        lookback_months: int   = 3,
        buffer_pct:      float = 10.0,
    ) -> dict:
        """
        Analyse past spending and suggest budget limits for a target month.

        How it works:
          1. Looks at actual expenses for the past N months per category
          2. Computes the average monthly spend per category
          3. Adds a buffer % on top (default 10%) to give breathing room
          4. Returns suggestions — LLM presents them to user for confirmation
          5. User confirms → call set_budgets_bulk() to apply them

        Parameters
        ──────────
        target_month     : the month to create budgets for  ('YYYY-MM')
        lookback_months  : how many past months to average  (default 3, max 12)
        buffer_pct       : % to add on top of average       (default 10%)
        """
        target = _month_start(target_month)
        lookback_months = min(max(1, lookback_months), 12)
        if not (0 <= buffer_pct <= 100):
            raise ValueError("'buffer_pct' must be between 0 and 100.")

        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(
                    """
                    SELECT
                        category,
                        subcategory,
                        AVG(monthly_total) AS avg_monthly,
                        MAX(monthly_total) AS max_monthly,
                        MIN(monthly_total) AS min_monthly,
                        COUNT(*)           AS months_seen
                    FROM (
                        SELECT
                            category,
                            subcategory,
                            DATE_FORMAT(date, '%%Y-%%m-01') AS month,
                            SUM(amount)                     AS monthly_total
                        FROM expenses
                        WHERE type     = 'expense'
                          AND category != ''
                          AND date >= DATE_FORMAT(
                                DATE_SUB(CURDATE(), INTERVAL %s MONTH), '%%Y-%%m-01'
                              )
                          AND DATE_FORMAT(date, '%%Y-%%m-01') < %s
                        GROUP BY category, subcategory, month
                    ) AS monthly_agg
                    GROUP BY category, subcategory
                    ORDER BY avg_monthly DESC
                    """,
                    (lookback_months, target),
                )
                rows = await cur.fetchall()

                # Check which budgets already exist for the target month
                await cur.execute(
                    "SELECT category, subcategory FROM budgets WHERE month = %s AND type = 'budget'",
                    (target,),
                )
                existing = {
                    (r["category"], r["subcategory"])
                    for r in await cur.fetchall()
                }

        suggestions = []
        for r in rows:
            avg       = float(r["avg_monthly"])
            suggested = round(avg * (1 + buffer_pct / 100), 2)
            key       = (r["category"], r["subcategory"])
            suggestions.append({
                "category":          r["category"],
                "subcategory":       r["subcategory"],
                "avg_monthly_spend": round(avg, 2),
                "max_monthly_spend": round(float(r["max_monthly"]), 2),
                "min_monthly_spend": round(float(r["min_monthly"]), 2),
                "months_of_data":    r["months_seen"],
                "suggested_budget":  suggested,
                "buffer_applied":    f"{buffer_pct}%",
                "already_set":       key in existing,
            })

        return {
            "target_month":    target,
            "lookback_months": lookback_months,
            "buffer_pct":      f"{buffer_pct}%",
            "suggestions":     suggestions,
            "next_step":       "Call set_budgets_bulk() with confirmed amounts to apply.",
        }


    # ── 12. COPY BUDGETS FROM MONTH ──────────────────────────────
    @mcp.tool()
    async def copy_budgets_from_month(
        from_month: str,
        to_month:   str,
        overwrite:  bool = False,
    ) -> dict:
        """
        Copy all spending budgets from one month to another.
        Useful for carrying forward last month's budgets into a new month.

        Parameters
        ──────────
        from_month : source month  ('YYYY-MM')
        to_month   : target month  ('YYYY-MM')
        overwrite  : if True, overwrite existing budgets in to_month
                     if False (default), skip categories already set in to_month
        """
        from_ms = _month_start(from_month)
        to_ms   = _month_start(to_month)

        if from_ms == to_ms:
            raise ValueError("'from_month' and 'to_month' cannot be the same.")

        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(
                    "SELECT category, subcategory, amount, note FROM budgets WHERE month = %s AND type = 'budget'",
                    (from_ms,),
                )
                source = list(await cur.fetchall())
                if not source:
                    raise ValueError(f"No budgets found for {from_ms} to copy from.")

                sql = (
                    """
                    INSERT INTO budgets (month, type, category, subcategory, amount, note)
                    VALUES (%s, 'budget', %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE amount = VALUES(amount), note = VALUES(note)
                    """
                    if overwrite else
                    """
                    INSERT IGNORE INTO budgets (month, type, category, subcategory, amount, note)
                    VALUES (%s, 'budget', %s, %s, %s, %s)
                    """
                )
                rows = [
                    (to_ms, r["category"], r["subcategory"], float(r["amount"]), r["note"])
                    for r in source
                ]
                await cur.executemany(sql, rows)
                copied = cur.rowcount

        return {
            "status":           "ok",
            "from_month":       from_ms,
            "to_month":         to_ms,
            "total_in_source":  len(source),
            "copied":           copied,
            "skipped":          len(source) - copied,
            "overwrite":        overwrite,
        }


    # ════════════════════════════════════════════════════════════
    # SAVINGS TOOLS
    # ════════════════════════════════════════════════════════════

    # ── 13. ADD SAVING ───────────────────────────────────────────
    @mcp.tool()
    async def add_saving(
        date:        str,
        amount:      float,
        category:    str = "General",
        subcategory: str = "",
        note:        str = "",
        tags:        str = "",
    ) -> dict:
        """
        Log a saving event — money you consciously saved today on something.

        Use this when the user says things like:
          "I saved 200 rupees today by cooking at home instead of ordering"
          "Saved 500 on transport by taking the bus instead of Uber"
          "Skipped coffee today, saved 150"

        Stored in the expenses table with type='saving'.
        NOT the same as setting a savings goal — this is the actual saving event.

        Parameters
        ──────────
        date        : YYYY-MM-DD — when the saving happened
        amount      : how much was saved  (>= 0)
        category    : what area  e.g. 'Food', 'Transport'
        subcategory : finer detail  e.g. 'Groceries', 'Fuel'
        note        : what you saved on and how
        tags        : comma-separated labels  e.g. 'frugal,lifestyle'
        """
        _validate_date(date)
        _validate_amount(amount)

        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO expenses
                        (date, type, amount, category, subcategory, note, tags)
                    VALUES (%s, 'saving', %s, %s, %s, %s, %s)
                    """,
                    (date, amount, category.strip(), subcategory.strip(), note.strip(), tags.strip()),
                )
                return {
                    "status":   "ok",
                    "id":       cur.lastrowid,
                    "date":     date,
                    "saved":    amount,
                    "category": category.strip(),
                    "note":     note.strip(),
                    "message":  f"Saving of {amount} logged under '{category.strip()}'.",
                }


    # ── 14. SET SAVINGS GOAL ─────────────────────────────────────
    @mcp.tool()
    async def set_savings_goal(
        month:       str,
        amount:      float,
        category:    str = "General",
        subcategory: str = "",
        note:        str = "",
    ) -> dict:
        """
        Set a monthly savings target for a category.
        Stored in the budgets table with type='savings_goal'.

        Use this when the user says things like:
          "I want to save 10000 rupees this month"
          "Set a savings goal of 2000 for Food this month"
          "I want to put aside 5000 for my vacation fund in March"

        Parameters
        ──────────
        month       : 'YYYY-MM' or 'YYYY-MM-DD'
        amount      : target savings amount  (>= 0)
        category    : savings bucket  e.g. 'General', 'Emergency Fund', 'Vacation'
        subcategory : optional detail
        note        : optional note about this goal
        """
        month = _month_start(month)
        _validate_amount(amount)

        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO budgets (month, type, category, subcategory, amount, note)
                    VALUES (%s, 'savings_goal', %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        amount = VALUES(amount),
                        note   = VALUES(note)
                    """,
                    (month, category.strip(), subcategory.strip(), amount, note.strip()),
                )
                action = "updated" if cur.rowcount == 2 else "created"
                return {
                    "status":   "ok",
                    "action":   action,
                    "month":    month,
                    "category": category.strip(),
                    "goal":     amount,
                    "note":     note.strip(),
                }


    # ── 15. GET SAVINGS SUMMARY ──────────────────────────────────
    @mcp.tool()
    async def get_savings_summary(
        month:    str,
        category: str | None = None,
    ) -> dict:
        """
        Compare actual savings logged vs savings goals set for a month.

        Returns per category:
          goal_amount      — target set via set_savings_goal()
          actual_saved     — sum of add_saving() entries for that month
          remaining        — how much more to save to hit the goal
          pct_achieved     — % of goal reached
          status           — GOAL MET / ON TRACK / PROGRESSING / BEHIND
          expected_by_now  — how much should have been saved by today based on days elapsed
          pace_status      — OVERSPENDING / ON PACE / UNDERSPENDING

        Also includes categories where savings were logged but no goal was set.

        Parameters
        ──────────
        month    : 'YYYY-MM' or 'YYYY-MM-DD'
        category : optional filter to a single savings category
        """
        ms = _month_start(month)
        me = _month_end(ms)
        ds = _days_stats(ms)

        cat_filter, params_extra = "", []
        if category:
            cat_filter = " AND b.category = %s"; params_extra.append(category)

        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:

                # Goals with actual savings joined
                await cur.execute(
                    f"""
                    SELECT
                        b.category,
                        b.subcategory,
                        b.amount                    AS goal_amount,
                        b.note                      AS goal_note,
                        COALESCE(SUM(e.amount), 0)  AS actual_saved
                    FROM budgets b
                    LEFT JOIN expenses e
                        ON  e.category    = b.category
                        AND (b.subcategory = '' OR e.subcategory = b.subcategory)
                        AND e.type        = 'saving'
                        AND e.date       >= %s
                        AND e.date        < %s
                    WHERE b.month = %s
                      AND b.type  = 'savings_goal'
                    {cat_filter}
                    GROUP BY b.category, b.subcategory, b.amount, b.note
                    ORDER BY b.category ASC
                    """,
                    [ms, me, ms] + params_extra,
                )
                with_goals = list(await cur.fetchall())

                # Savings logged with no goal set
                e_cat_filter = cat_filter.replace("b.category", "e.category") if cat_filter else ""
                await cur.execute(
                    f"""
                    SELECT
                        e.category,
                        e.subcategory,
                        0               AS goal_amount,
                        ''              AS goal_note,
                        SUM(e.amount)   AS actual_saved
                    FROM expenses e
                    LEFT JOIN budgets b
                        ON  b.category = e.category
                        AND b.month    = %s
                        AND b.type     = 'savings_goal'
                    WHERE e.type  = 'saving'
                      AND e.date >= %s
                      AND e.date  < %s
                      AND b.id IS NULL
                    {e_cat_filter}
                    GROUP BY e.category, e.subcategory
                    ORDER BY actual_saved DESC
                    """,
                    [ms, ms, me] + params_extra,
                )
                without_goals = list(await cur.fetchall())

        all_rows    = with_goals + without_goals
        total_goal  = sum(float(r["goal_amount"])  for r in all_rows)
        total_saved = sum(float(r["actual_saved"]) for r in all_rows)

        breakdown = []
        for r in all_rows:
            goal     = float(r["goal_amount"])
            actual   = float(r["actual_saved"])
            pct      = round(actual / goal * 100, 2) if goal else None
            expected = round(goal * ds["days_elapsed"] / ds["days_in_month"], 2) if goal else None
            breakdown.append({
                "category":        r["category"],
                "subcategory":     r["subcategory"],
                "goal_amount":     goal,
                "goal_note":       r.get("goal_note", ""),
                "actual_saved":    round(actual, 2),
                "remaining":       round(goal - actual, 2) if goal else None,
                "pct_achieved":    f"{pct}%" if pct is not None else "No goal set",
                "status":          _savings_status(pct) if pct is not None else "NO GOAL",
                "expected_by_now": expected,
                "pace_status":     _pace_label(actual, expected) if expected else "N/A",
            })

        return {
            "month":         ms,
            "total_goal":    round(total_goal, 2),
            "total_saved":   round(total_saved, 2),
            "remaining":     round(total_goal - total_saved, 2),
            "overall_pct":   f"{round(total_saved / total_goal * 100, 2)}%" if total_goal else "N/A",
            "days_elapsed":  ds["days_elapsed"],
            "days_in_month": ds["days_in_month"],
            "breakdown":     breakdown,
        }


    # ── 16. GET SAVINGS TREND ────────────────────────────────────
    @mcp.tool()
    async def get_savings_trend(
        months:   int        = 6,
        category: str | None = None,
    ) -> list[dict]:
        """
        Month-over-month savings history — how much was actually saved each month
        vs the savings goal set for that month.

        Parameters
        ──────────
        months   : how many past months to include  (default 6, max 24)
        category : optional — filter to a single savings category
        """
        months = min(max(1, months), 24)
        cat_filter, params_extra = "", []
        if category:
            cat_filter = " AND category = %s"; params_extra.append(category)

        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:

                # Actual savings per month
                await cur.execute(
                    f"""
                    SELECT
                        DATE_FORMAT(date, '%%Y-%%m-01') AS month,
                        SUM(amount)                     AS total_saved,
                        COUNT(*)                        AS saving_events
                    FROM expenses
                    WHERE type  = 'saving'
                      AND date >= DATE_FORMAT(
                            DATE_SUB(CURDATE(), INTERVAL %s MONTH), '%%Y-%%m-01'
                          )
                    {cat_filter}
                    GROUP BY month
                    ORDER BY month ASC
                    """,
                    [months] + params_extra,
                )
                savings_rows = {r["month"]: r for r in await cur.fetchall()}

                # Goals per month
                await cur.execute(
                    f"""
                    SELECT
                        month,
                        SUM(amount) AS total_goal
                    FROM budgets
                    WHERE type  = 'savings_goal'
                      AND month >= DATE_FORMAT(
                            DATE_SUB(CURDATE(), INTERVAL %s MONTH), '%%Y-%%m-01'
                          )
                    {cat_filter}
                    GROUP BY month
                    ORDER BY month ASC
                    """,
                    [months] + params_extra,
                )
                goal_rows = {
                    (r["month"].isoformat() if hasattr(r["month"], "isoformat") else str(r["month"])): r
                    for r in await cur.fetchall()
                }

        all_months = sorted(set(list(savings_rows.keys()) + list(goal_rows.keys())))

        result = []
        for m in all_months:
            m_str  = m if isinstance(m, str) else m.isoformat()
            saved  = float(savings_rows[m]["total_saved"])  if m in savings_rows else 0.0
            goal   = float(goal_rows[m_str]["total_goal"])  if m_str in goal_rows else 0.0
            pct    = round(saved / goal * 100, 2)           if goal else None
            result.append({
                "month":         m_str,
                "total_goal":    round(goal, 2),
                "total_saved":   round(saved, 2),
                "difference":    round(goal - saved, 2),
                "pct_achieved":  f"{pct}%" if pct is not None else "No goal set",
                "status":        _savings_status(pct) if pct is not None else "NO GOAL",
                "saving_events": savings_rows[m]["saving_events"] if m in savings_rows else 0,
            })

        return result