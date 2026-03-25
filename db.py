"""
db.py
─────
Shared async connection pool for the Expense Tracker MCP server.
Pool is initialised lazily on first use — no startup hook required.
"""

from __future__ import annotations

import os
import aiomysql
from dotenv import load_dotenv

load_dotenv()

_pool: aiomysql.Pool | None = None


def _db_config() -> dict:
    return {
        "host":       os.getenv("MYSQL_HOST", "localhost"),
        "port":       int(os.getenv("MYSQL_PORT", 3306)),
        "user":       os.getenv("MYSQL_USER", "root"),
        "password":   os.getenv("MYSQL_PASSWORD"),
        "db":         os.getenv("MYSQL_DB", "mydb"),
        "autocommit": True,
        "charset":    "utf8mb4",
    }


async def init_pool(minsize: int = 2, maxsize: int = 10) -> None:
    global _pool
    if _pool is not None:
        return

    _pool = await aiomysql.create_pool(
        minsize=minsize,
        maxsize=maxsize,
        **_db_config(),
    )

    async with _pool.acquire() as conn:
        async with conn.cursor() as cur:

            await cur.execute("""
                CREATE TABLE IF NOT EXISTS expenses (
                    id              INT AUTO_INCREMENT PRIMARY KEY,
                    date            DATE          NOT NULL,
                    type            ENUM('expense','income','saving') NOT NULL DEFAULT 'expense',
                    amount          DECIMAL(10,2) NOT NULL CHECK (amount >= 0),
                    category        VARCHAR(100)  NOT NULL DEFAULT '',
                    subcategory     VARCHAR(100)  NOT NULL DEFAULT '',
                    note            VARCHAR(255)  NOT NULL DEFAULT '',
                    payment_method  VARCHAR(50)   NOT NULL DEFAULT '',
                    recurring       BOOLEAN       NOT NULL DEFAULT FALSE,
                    tags            VARCHAR(255)  NOT NULL DEFAULT '',
                    created_at      TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    INDEX  idx_date          (date),
                    INDEX  idx_type          (type),
                    INDEX  idx_category      (category),
                    INDEX  idx_subcategory   (subcategory),
                    INDEX  idx_payment       (payment_method),
                    INDEX  idx_type_date     (type, date),
                    INDEX  idx_cat_type_date (category, type, date),
                    FULLTEXT INDEX ft_search (note, tags)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)

            await cur.execute("""
                CREATE TABLE IF NOT EXISTS budgets (
                    id          INT AUTO_INCREMENT PRIMARY KEY,
                    month       DATE          NOT NULL,
                    type        ENUM('budget','savings_goal') NOT NULL DEFAULT 'budget',
                    category    VARCHAR(100)  NOT NULL DEFAULT '',
                    subcategory VARCHAR(100)  NOT NULL DEFAULT '',
                    amount      DECIMAL(10,2) NOT NULL CHECK (amount >= 0),
                    note        VARCHAR(255)  NOT NULL DEFAULT '',
                    created_at  TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY uq_budget      (month, type, category, subcategory),
                    INDEX      idx_month      (month),
                    INDEX      idx_type       (type),
                    INDEX      idx_category   (category),
                    INDEX      idx_month_type (month, type)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)


async def get_pool() -> aiomysql.Pool:
    """Return the pool, initialising it lazily on first call."""
    if _pool is None:
        await init_pool()
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        await _pool.wait_closed()
        _pool = None