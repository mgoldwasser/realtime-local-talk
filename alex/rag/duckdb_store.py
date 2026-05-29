"""DuckDB-backed structured retrieval for TAA committee data.

Tables (see ingest_cli/datapoints.py for the synthetic seed data):

    current_allocations   asset_class, region, policy_bps, current_bps,
                          active_bps, as_of_date
    time_series_returns   ticker, date, total_return_bps, close
    factor_panel          factor, current_z, six_month_avg_z, comment
    economic_indicators   indicator, date, value, change_bps

Safety:
- Read-only connection (no DDL/DML allowed even if the LLM tries)
- ``LIMIT`` enforced if the LLM forgets one
- Query timeout
- Schema is exposed to the LLM so it can construct meaningful queries
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
from loguru import logger


DB_SCHEMA_DESCRIPTION = """\
Tables available (DuckDB SQL):

current_allocations
  asset_class    text   e.g. 'US Equity', 'International Equity', 'EM Equity', 'US Treasury', 'IG Credit', 'HY Credit', 'Cash'
  region         text   e.g. 'United States', 'Developed ex-US', 'Emerging', 'Global'
  policy_bps     int    target weight in basis points (10000 = 100%)
  current_bps    int    current weight in basis points
  active_bps     int    current_bps - policy_bps (positive = overweight)
  as_of_date     date

time_series_returns
  ticker         text   e.g. 'SPY', 'IWM', 'EFA', 'EEM', 'AGG', 'TLT', 'HYG', 'LQD'
  date           date
  total_return_bps int  daily total return in basis points
  close          double price level at close

factor_panel
  factor         text   'Quality', 'Low Volatility', 'Value', 'Size', 'Momentum'
  as_of_date     date
  current_z      double current factor z-score vs Russell 3000
  six_month_avg_z double rolling 6m avg
  comment        text

economic_indicators
  indicator      text   e.g. 'Fed Funds Rate', '10Y Treasury', 'CPI YoY', 'PCE Core'
  date           date
  value          double
  change_bps     int    change from prior period

Always:
- Use the exact column names above (DuckDB is case-insensitive)
- Quote dates in YYYY-MM-DD format
- LIMIT 20 unless explicitly answering a single-row question
"""


@dataclass
class SqlResult:
    columns: list[str]
    rows: list[tuple]
    error: str | None = None

    def to_text(self, max_rows: int = 20) -> str:
        if self.error:
            return f"SQL error: {self.error}"
        if not self.rows:
            return "(no rows)"
        rows = self.rows[:max_rows]
        # Compact tabular text the LLM can read.
        lines = [" | ".join(self.columns)]
        lines.append(" | ".join(["---"] * len(self.columns)))
        for r in rows:
            lines.append(" | ".join(str(v) for v in r))
        suffix = f"\n({len(self.rows)} rows total)" if len(self.rows) > max_rows else ""
        return "\n".join(lines) + suffix


class DuckDBCorpus:
    """Read-only wrapper over a DuckDB file."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        # `read_only=True` makes the LLM physically unable to mutate state.
        self._conn = duckdb.connect(str(self.db_path), read_only=True)

    def query(self, sql: str, *, row_limit: int = 200) -> SqlResult:
        """Execute a SELECT. Returns rows or an error message — never raises."""
        try:
            cur = self._conn.execute(sql)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchmany(row_limit)
            return SqlResult(columns=cols, rows=[tuple(r) for r in rows])
        except Exception as e:
            return SqlResult(columns=[], rows=[], error=str(e))

    def empty(self) -> bool:
        try:
            r = self._conn.execute(
                "SELECT count(*) FROM information_schema.tables "
                "WHERE table_schema = 'main'"
            ).fetchone()
            return (r[0] or 0) == 0
        except Exception:
            return True

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
