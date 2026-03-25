"""
server.py
─────────
Entry point for the Expense Tracker MCP server.

Usage
─────
  python server.py                              # stdio (default)
  python server.py --transport sse --port 8000  # SSE
"""

from __future__ import annotations

import argparse

from fastmcp import FastMCP

from db import init_pool          # called lazily via get_pool()
import expenses as expenses_module
import budgets  as budgets_module


# ─────────────────────────────────────────────
# CREATE THE MCP INSTANCE  (no lifespan kwarg)
# ─────────────────────────────────────────────

mcp = FastMCP("expense-tracker")


# ─────────────────────────────────────────────
# REGISTER TOOL MODULES
# ─────────────────────────────────────────────

expenses_module.register(mcp)
budgets_module.register(mcp)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="expense-tracker-mcp")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        mcp.run(transport="stdio")