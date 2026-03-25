"""
expenses.py
───────────
All expense-related MCP tools for the Expense Tracker server.
Registered onto the FastMCP instance passed in from server.py via register(mcp).
"""

from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import aiomysql

from db import get_pool

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _clean(rows: list[dict] | dict) -> Any:
    """
    Recursively convert Decimal → float and date/datetime → ISO string
    so every response is JSON-serialisable.
    """
    if isinstance(rows, list):
        return [_clean(r) for r in rows]
    if isinstance(rows, dict):
        return {
            k: (
                float(v)        if isinstance(v, Decimal)          else
                v.isoformat()   if isinstance(v, (date, datetime)) else
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


def _validate_type(t: str) -> None:
    allowed = {"expense", "income", "saving"}
    if t not in allowed:
        raise ValueError(f"'type' must be one of {allowed}, got: {t!r}")


def _validate_amount(a: float) -> None:
    if a < 0:
        raise ValueError(f"'amount' must be ≥ 0, got: {a}")


def _build_where(
    *,
    start_date: str | None   = None,
    end_date:   str | None   = None,
    type_:      str | None   = None,
    category:   str | None   = None,
    subcategory:str | None   = None,
    payment_method: str | None = None,
    recurring:  bool | None  = None,
    tags:       str | None   = None,
) -> tuple[str, list]:
    """Build a reusable WHERE clause + params list from optional filters."""
    clauses, params = [], []

    if start_date:
        clauses.append("date >= %s"); params.append(start_date)
    if end_date:
        clauses.append("date <= %s"); params.append(end_date)
    if type_:
        clauses.append("type = %s"); params.append(type_)
    if category:
        clauses.append("category = %s"); params.append(category)
    if subcategory:
        clauses.append("subcategory = %s"); params.append(subcategory)
    if payment_method:
        clauses.append("payment_method = %s"); params.append(payment_method)
    if recurring is not None:
        clauses.append("recurring = %s"); params.append(int(recurring))
    if tags:
        clauses.append("FIND_IN_SET(%s, tags) > 0"); params.append(tags)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


# ─────────────────────────────────────────────
# REGISTRATION ENTRY POINT
# ─────────────────────────────────────────────

def register(mcp) -> None:
    """Mount all expense tools onto the FastMCP instance."""

    # ════════════════════════════════════════════
    # 1. ADD SINGLE EXPENSE
    # ════════════════════════════════════════════
    @mcp.tool()
    async def add_expense(
        date:           str,
        amount:         float,
        type:           str  = "expense",
        category:       str  = "",
        subcategory:    str  = "",
        note:           str  = "",
        payment_method: str  = "",
        recurring:      bool = False,
        tags:           str  = "",
    ) -> dict:
        """
        Add a single expense / income / saving entry.

        Parameters
        ──────────
        date           : YYYY-MM-DD
        amount         : positive number
        type           : 'expense' | 'income' | 'saving'
        category       : e.g. 'Food', 'Transport'
        subcategory    : e.g. 'Groceries', 'Uber'
        note           : free-text description
        payment_method : 'Cash' | 'UPI' | 'Card' | 'Net Banking' | …
        recurring      : True if this repeats every month
        tags           : comma-separated labels, e.g. 'work,reimbursable'
        """
        _validate_date(date)
        _validate_type(type)
        _validate_amount(amount)

        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO expenses
                        (date, type, amount, category, subcategory,
                         note, payment_method, recurring, tags)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (date, type, amount, category, subcategory,
                     note, payment_method, int(recurring), tags),
                )
                return {"status": "ok", "id": cur.lastrowid}


    # ════════════════════════════════════════════
    # 2. ADD BULK EXPENSES
    # ════════════════════════════════════════════
    @mcp.tool()
    async def add_expenses_bulk(entries: list[dict]) -> dict:
        """
        Insert multiple expense entries in a single transaction.

        Each entry in the list follows the same schema as add_expense.
        Example:
            [
              {"date":"2025-03-01","amount":150,"category":"Food","type":"expense"},
              {"date":"2025-03-01","amount":5000,"category":"Salary","type":"income"}
            ]
        """
        if not entries:
            raise ValueError("'entries' list is empty.")

        rows = []
        for i, e in enumerate(entries):
            d    = e.get("date", "")
            amt  = float(e.get("amount", -1))
            typ  = e.get("type", "expense")
            _validate_date(d,   f"entries[{i}].date")
            _validate_type(typ)
            _validate_amount(amt)
            rows.append((
                d, typ, amt,
                e.get("category",       ""),
                e.get("subcategory",    ""),
                e.get("note",           ""),
                e.get("payment_method", ""),
                int(e.get("recurring",  False)),
                e.get("tags",           ""),
            ))

        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.executemany(
                    """
                    INSERT INTO expenses
                        (date, type, amount, category, subcategory,
                         note, payment_method, recurring, tags)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    rows,
                )
                return {"status": "ok", "inserted": len(rows)}


    # ════════════════════════════════════════════
    # 3. GET EXPENSE BY ID
    # ════════════════════════════════════════════
    @mcp.tool()
    async def get_expense_by_id(id: int) -> dict:
        """Fetch a single expense row by its primary-key ID."""
        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(
                    "SELECT * FROM expenses WHERE id = %s", (id,)
                )
                row = await cur.fetchone()

        if row is None:
            raise ValueError(f"No expense found with id={id}")
        return _clean(row)


    # ════════════════════════════════════════════
    # 4. LIST EXPENSES  (paginated + filtered)
    # ════════════════════════════════════════════
    @mcp.tool()
    async def list_expenses(
        start_date:     str | None  = None,
        end_date:       str | None  = None,
        type:           str | None  = None,
        category:       str | None  = None,
        subcategory:    str | None  = None,
        payment_method: str | None  = None,
        recurring:      bool | None = None,
        tags:           str | None  = None,
        order_by:       str         = "date",
        order_dir:      str         = "DESC",
        limit:          int         = 50,
        offset:         int         = 0,
    ) -> dict:
        """
        List expenses with optional filters and pagination.

        Filters
        ───────
        start_date / end_date  : YYYY-MM-DD range (inclusive)
        type                   : 'expense' | 'income' | 'saving'
        category / subcategory : exact match
        payment_method         : exact match
        recurring              : True | False
        tags                   : single tag to match (FIND_IN_SET)

        Pagination
        ──────────
        order_by  : column name (date | amount | category | created_at)
        order_dir : ASC | DESC
        limit     : rows per page (max 200)
        offset    : skip N rows
        """
        # Guard inputs
        if start_date: _validate_date(start_date, "start_date")
        if end_date:   _validate_date(end_date,   "end_date")
        if type:       _validate_type(type)

        allowed_cols = {"date", "amount", "category", "created_at", "id"}
        if order_by not in allowed_cols:
            raise ValueError(f"order_by must be one of {allowed_cols}")
        if order_dir.upper() not in {"ASC", "DESC"}:
            raise ValueError("order_dir must be ASC or DESC")
        limit = min(max(1, limit), 200)

        where, params = _build_where(
            start_date=start_date, end_date=end_date, type_=type,
            category=category, subcategory=subcategory,
            payment_method=payment_method, recurring=recurring, tags=tags,
        )

        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                # total count
                await cur.execute(
                    f"SELECT COUNT(*) AS total FROM expenses {where}", params
                )
                total = (await cur.fetchone())["total"]

                # paginated rows
                await cur.execute(
                    f"""
                    SELECT * FROM expenses
                    {where}
                    ORDER BY {order_by} {order_dir.upper()}
                    LIMIT %s OFFSET %s
                    """,
                    params + [limit, offset],
                )
                rows = await cur.fetchall()

        return {
            "total":  total,
            "limit":  limit,
            "offset": offset,
            "data":   _clean(list(rows)),
        }


    # ════════════════════════════════════════════
    # 5. SEARCH EXPENSES  (fulltext)
    # ════════════════════════════════════════════
    @mcp.tool()
    async def search_expenses(
        keyword:    str,
        start_date: str | None = None,
        end_date:   str | None = None,
        limit:      int        = 50,
    ) -> dict:
        """
        Full-text keyword search across note, tags, category, and subcategory.

        Uses MySQL FULLTEXT index on (note, tags) for speed, plus a LIKE
        fallback on category and subcategory columns.

        Parameters
        ──────────
        keyword    : word or phrase to search
        start_date : optional YYYY-MM-DD lower bound
        end_date   : optional YYYY-MM-DD upper bound
        limit      : max results (default 50, max 200)
        """
        if not keyword or not keyword.strip():
            raise ValueError("'keyword' cannot be empty.")
        if start_date: _validate_date(start_date, "start_date")
        if end_date:   _validate_date(end_date,   "end_date")
        limit = min(max(1, limit), 200)

        like = f"%{keyword}%"
        date_clause, date_params = "", []
        if start_date: date_clause += " AND date >= %s"; date_params.append(start_date)
        if end_date:   date_clause += " AND date <= %s"; date_params.append(end_date)

        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(
                    f"""
                    SELECT *, MATCH(note, tags) AGAINST (%s) AS relevance
                    FROM expenses
                    WHERE (
                        MATCH(note, tags) AGAINST (%s IN BOOLEAN MODE)
                        OR category    LIKE %s
                        OR subcategory LIKE %s
                    )
                    {date_clause}
                    ORDER BY relevance DESC, date DESC
                    LIMIT %s
                    """,
                    [keyword, keyword, like, like] + date_params + [limit],
                )
                rows = await cur.fetchall()

        return {"keyword": keyword, "count": len(rows), "data": _clean(list(rows))}


    # ════════════════════════════════════════════
    # 6. SUMMARIZE BY CATEGORY
    # ════════════════════════════════════════════
    @mcp.tool()
    async def summarize_by_category(
        start_date:  str,
        end_date:    str,
        t:        str | None = None,
        subcategory: bool       = False,
    ) -> list[dict]:
        """
        Total amount grouped by category (and optionally subcategory)
        for a given date range.

        Parameters
        ──────────
        start_date   : YYYY-MM-DD
        end_date     : YYYY-MM-DD
        type         : filter to 'expense' | 'income' | 'saving'  (optional)
        subcategory  : if True, also group by subcategory
        """
        _validate_date(start_date, "start_date")
        _validate_date(end_date,   "end_date")
        if t: _validate_type(t)

        group_cols = "category, subcategory" if subcategory else "category"
        type_clause = "AND type = %s" if t else ""
        params = [start_date, end_date] + ([t] if t else [])

        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(
                    f"""
                    SELECT {group_cols},
                           SUM(amount)  AS total_amount,
                           COUNT(*)     AS num_transactions,
                           AVG(amount)  AS avg_amount,
                           MAX(amount)  AS max_amount,
                           MIN(amount)  AS min_amount
                    FROM expenses
                    WHERE date BETWEEN %s AND %s
                      AND amount > 0
                      {type_clause}
                    GROUP BY {group_cols}
                    ORDER BY total_amount DESC
                    """,
                    params,
                )
                rows = await cur.fetchall()
        return _clean(list(rows))


    # ════════════════════════════════════════════
    # 7. SUMMARIZE BY PERIOD
    # ════════════════════════════════════════════
    @mcp.tool()
    async def summarize_by_period(
        start_date: str,
        end_date:   str,
        period:     str         = "month",
        type:       str | None  = None,
        category:   str | None  = None,
    ) -> list[dict]:
        """
        Aggregate expenses over daily / weekly / monthly buckets.

        Parameters
        ──────────
        start_date : YYYY-MM-DD
        end_date   : YYYY-MM-DD
        period     : 'day' | 'week' | 'month'
        type       : optional filter
        category   : optional category filter
        """
        _validate_date(start_date, "start_date")
        _validate_date(end_date,   "end_date")
        if type: _validate_type(type)

        period_expr = {
            "day":   "DATE(date)",
            "week":  "DATE(DATE_SUB(date, INTERVAL WEEKDAY(date) DAY))",
            "month": "DATE_FORMAT(date, '%Y-%m-01')",
        }.get(period)
        if not period_expr:
            raise ValueError("'period' must be 'day', 'week', or 'month'.")

        extra_clauses, params = [], [start_date, end_date]
        if type:     extra_clauses.append("AND type = %s");     params.append(type)
        if category: extra_clauses.append("AND category = %s"); params.append(category)
        extra = " ".join(extra_clauses)

        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(
                    f"""
                    SELECT
                        {period_expr}   AS period_start,
                        type,
                        SUM(amount)     AS total_amount,
                        COUNT(*)        AS num_transactions
                    FROM expenses
                    WHERE date BETWEEN %s AND %s {extra}
                    GROUP BY period_start, type
                    ORDER BY period_start ASC, type
                    """,
                    params,
                )
                rows = await cur.fetchall()
        return _clean(list(rows))


    # ════════════════════════════════════════════
    # 8. GET TOP EXPENSES
    # ════════════════════════════════════════════
    @mcp.tool()
    async def get_top_expenses(
        start_date: str,
        end_date:   str,
        top_n:      int        = 10,
        type:       str        = "expense",
        category:   str | None = None,
    ) -> list[dict]:
        """
        Return the top N largest single transactions in a date range.

        Parameters
        ──────────
        start_date : YYYY-MM-DD
        end_date   : YYYY-MM-DD
        top_n      : number of results (default 10, max 100)
        type       : 'expense' | 'income' | 'saving'
        category   : optional category filter
        """
        _validate_date(start_date, "start_date")
        _validate_date(end_date,   "end_date")
        _validate_type(type)
        top_n = min(max(1, top_n), 100)

        extra, params = "", [start_date, end_date, type]
        if category:
            extra = "AND category = %s"; params.append(category)

        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(
                    f"""
                    SELECT id, date, type, amount, category,
                           subcategory, note, payment_method, tags
                    FROM expenses
                    WHERE date BETWEEN %s AND %s
                      AND type = %s {extra}
                    ORDER BY amount DESC
                    LIMIT %s
                    """,
                    params + [top_n],
                )
                rows = await cur.fetchall()
        return _clean(list(rows))


    # ════════════════════════════════════════════
    # 9. GET CASHFLOW
    # ════════════════════════════════════════════
    @mcp.tool()
    async def get_cashflow(start_date: str, end_date: str) -> dict:
        """
        Net cashflow summary for a date range.

        Returns total income, total expenses, total savings,
        net (income − expenses), and savings rate %.
        """
        _validate_date(start_date, "start_date")
        _validate_date(end_date,   "end_date")

        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(
                    """
                    SELECT
                        type,
                        SUM(amount)  AS total,
                        COUNT(*)     AS transactions
                    FROM expenses
                    WHERE date BETWEEN %s AND %s
                    GROUP BY type
                    """,
                    (start_date, end_date),
                )
                rows = await cur.fetchall()

        summary = {r["type"]: _clean(r) for r in rows}
        income   = float(summary.get("income",  {}).get("total", 0))
        expense  = float(summary.get("expense", {}).get("total", 0))
        saving   = float(summary.get("saving",  {}).get("total", 0))
        net      = round(income - expense, 2)
        savings_rate = round((saving / income * 100), 2) if income else 0.0

        return {
            "start_date":    start_date,
            "end_date":      end_date,
            "total_income":  income,
            "total_expense": expense,
            "total_saving":  saving,
            "net":           net,
            "savings_rate":  f"{savings_rate}%",
            "breakdown":     list(summary.values()),
        }


    # ════════════════════════════════════════════
    # 10. MONTHLY TREND
    # ════════════════════════════════════════════
    @mcp.tool()
    async def get_monthly_trend(
        category:   str,
        months:     int = 6,
        type:       str = "expense",
    ) -> list[dict]:
        """
        Month-over-month spending trend for a specific category.

        Parameters
        ──────────
        category : category name to track
        months   : how many past months to include (default 6, max 24)
        type     : 'expense' | 'income' | 'saving'
        """
        _validate_type(type)
        months = min(max(1, months), 24)

        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(
                    """
                    SELECT
                        DATE_FORMAT(date, '%Y-%m-01') AS month,
                        SUM(amount)  AS total_amount,
                        COUNT(*)     AS num_transactions,
                        AVG(amount)  AS avg_amount
                    FROM expenses
                    WHERE category = %s
                      AND type     = %s
                      AND date >= DATE_SUB(CURDATE(), INTERVAL %s MONTH)
                    GROUP BY month
                    ORDER BY month ASC
                    """,
                    (category, type, months),
                )
                rows = await cur.fetchall()
        return _clean(list(rows))


    # ════════════════════════════════════════════
    # 11. UPDATE EXPENSE
    # ════════════════════════════════════════════
    @mcp.tool()
    async def update_expense(
            id: int,
            date: str | None = None,
            type: str | None = None,
            amount: float | None = None,
            category: str | None = None,
            subcategory: str | None = None,
            note: str | None = None,
            payment_method: str | None = None,
            recurring: bool | None = None,
            tags: str | None = None,
    ) -> dict:
        """
        Update any fields of an expense entry by ID.
        Pass only the fields you want to change — all are optional except id.

        Updatable fields
        ────────────────
        date, type, amount, category, subcategory,
        note, payment_method, recurring, tags
        """
        allowed = {
            "date": date, "type": type, "amount": amount,
            "category": category, "subcategory": subcategory,
            "note": note, "payment_method": payment_method,
            "recurring": recurring, "tags": tags,
        }
        updates = {k: v for k, v in allowed.items() if v is not None}

        if not updates:
            raise ValueError("No valid fields provided to update.")

        if "date" in updates: _validate_date(updates["date"])
        if "type" in updates: _validate_type(updates["type"])
        if "amount" in updates: _validate_amount(float(updates["amount"]))
        if "recurring" in updates:
            updates["recurring"] = int(bool(updates["recurring"]))

        set_clause = ", ".join(f"{col} = %s" for col in updates)
        params = list(updates.values()) + [id]

        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"UPDATE expenses SET {set_clause} WHERE id = %s", params
                )
                if cur.rowcount == 0:
                    raise ValueError(f"No expense found with id={id}")

        return {"status": "ok", "updated_id": id, "fields_changed": list(updates.keys())}

    # ════════════════════════════════════════════
    # 12. DELETE BY ID
    # ════════════════════════════════════════════
    @mcp.tool()
    async def delete_expense_by_id(id: int) -> dict:
        """Delete a single expense entry by its ID."""
        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM expenses WHERE id = %s", (id,)
                )
                if cur.rowcount == 0:
                    raise ValueError(f"No expense found with id={id}")

        return {"status": "ok", "deleted_id": id}


    # ════════════════════════════════════════════
    # 13. DELETE BY FILTER (bulk)
    # ════════════════════════════════════════════
    @mcp.tool()
    async def delete_expenses_by_filter(
        start_date:  str | None = None,
        end_date:    str | None = None,
        type:        str | None = None,
        category:    str | None = None,
        subcategory: str | None = None,
    ) -> dict:
        """
        Bulk-delete expenses matching the given filters.
        At least one filter must be provided to prevent accidental wipe.

        Parameters
        ──────────
        start_date / end_date : date range (YYYY-MM-DD)
        type                  : 'expense' | 'income' | 'saving'
        category              : exact match
        subcategory           : exact match
        """
        if not any([start_date, end_date, type, category, subcategory]):
            raise ValueError(
                "At least one filter is required for bulk delete. "
                "Use delete_expense_by_id() to delete a single row."
            )

        if start_date: _validate_date(start_date, "start_date")
        if end_date:   _validate_date(end_date,   "end_date")
        if type:       _validate_type(type)

        where, params = _build_where(
            start_date=start_date, end_date=end_date, type_=type,
            category=category, subcategory=subcategory,
        )

        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"DELETE FROM expenses {where}", params
                )
                deleted = cur.rowcount

        return {"status": "ok", "deleted_rows": deleted}


    # ════════════════════════════════════════════
    # 14. GET CATEGORIES
    # ════════════════════════════════════════════
    @mcp.tool()
    async def get_categories() -> list[dict]:
        """
        Return all distinct category + subcategory combinations
        ever recorded, ordered alphabetically.

        Useful for the LLM to suggest valid categories to the user
        and avoid creating duplicates with slightly different spellings.
        """
        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(
                    """
                    SELECT
                        category,
                        subcategory,
                        COUNT(*)        AS usage_count,
                        MAX(date)       AS last_used
                    FROM expenses
                    WHERE category != ''
                    GROUP BY category, subcategory
                    ORDER BY category ASC, subcategory ASC
                    """
                )
                rows = await cur.fetchall()
        return _clean(list(rows))