"""Seed synthetic TAA data into the DuckDB store.

CLI:
    uv run python -m ingest_cli.datapoints seed [--db data/duck.db] [--fresh]

Generates plausible-looking but FAKE numbers — never substitute for real data.
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path

import duckdb
import typer
from loguru import logger

cli = typer.Typer(no_args_is_help=True, add_completion=False)

ASSET_CLASSES = [
    # (asset_class, region, policy_bps, active_bps)
    ("US Equity", "United States", 3500, -75),       # underweight US equity
    ("International Equity", "Developed ex-US", 1500, 25),
    ("EM Equity", "Emerging", 800, 100),             # overweight EM (matches the note)
    ("US Treasury", "United States", 2200, 150),     # duration extension
    ("IG Credit", "United States", 700, -50),
    ("HY Credit", "United States", 400, -75),        # underweight HY (matches note)
    ("Cash", "Global", 100, -100),
    ("Commodities", "Global", 300, 25),
    ("REITs", "Global", 500, 0),
]

TICKERS = ["SPY", "IWM", "EFA", "EEM", "AGG", "TLT", "HYG", "LQD", "GLD"]

FACTORS = [
    ("Quality", 0.42, 0.31, "Driven by IG balance-sheet tilts; consistent with late-cycle stance."),
    ("Low Volatility", 0.28, 0.19, "Healthcare and utilities OW; defensive intent approved Oct '25."),
    ("Value", -0.04, -0.11, "Roughly neutral; small drift toward growth this quarter."),
    ("Size", -0.34, -0.29, "Persistent large-cap bias; no near-term plan to close."),
    ("Momentum", -0.22, -0.05, "Trimmed AI mega-cap winners; mean-reversion is the live risk."),
]

INDICATORS = [
    ("Fed Funds Rate", 4.375, -25),
    ("10Y Treasury", 4.18, -4),
    ("CPI YoY", 2.9, -10),
    ("PCE Core", 2.7, -8),
    ("Unemployment Rate", 4.1, 0),
    ("DXY", 102.4, -35),
]


@cli.command()
def seed(
    db: Path = typer.Option(Path("data/duck.db"), "--db"),
    fresh: bool = typer.Option(False, "--fresh"),
    seed_int: int = typer.Option(42, "--seed", help="RNG seed for reproducibility."),
) -> None:
    """Create the DuckDB and seed fake TAA data."""
    db.parent.mkdir(parents=True, exist_ok=True)
    if fresh and db.exists():
        db.unlink()
        logger.info(f"removed {db}")

    rng = random.Random(seed_int)
    conn = duckdb.connect(str(db), read_only=False)
    try:
        today = date(2026, 2, 28)

        # current_allocations -----------------------------------------------
        conn.execute("DROP TABLE IF EXISTS current_allocations")
        conn.execute(
            """
            CREATE TABLE current_allocations (
                asset_class TEXT,
                region TEXT,
                policy_bps INTEGER,
                current_bps INTEGER,
                active_bps INTEGER,
                as_of_date DATE
            )
            """
        )
        for ac, region, policy, active in ASSET_CLASSES:
            conn.execute(
                "INSERT INTO current_allocations VALUES (?, ?, ?, ?, ?, ?)",
                [ac, region, policy, policy + active, active, today],
            )

        # time_series_returns: 6 months of daily returns per ticker --------
        conn.execute("DROP TABLE IF EXISTS time_series_returns")
        conn.execute(
            """
            CREATE TABLE time_series_returns (
                ticker TEXT,
                date DATE,
                total_return_bps INTEGER,
                close DOUBLE
            )
            """
        )
        start_close = {
            "SPY": 510.0, "IWM": 210.0, "EFA": 80.0, "EEM": 42.0,
            "AGG": 100.0, "TLT": 90.0, "HYG": 78.0, "LQD": 108.0, "GLD": 195.0,
        }
        n_days = 126
        for t in TICKERS:
            close = start_close[t]
            for d in range(n_days):
                day = today - timedelta(days=n_days - d)
                # Equities more volatile than bonds.
                vol = 80 if t in {"SPY", "IWM", "EFA", "EEM", "GLD"} else 25
                ret_bps = int(rng.gauss(2, vol))
                close *= 1 + ret_bps / 10000
                conn.execute(
                    "INSERT INTO time_series_returns VALUES (?, ?, ?, ?)",
                    [t, day, ret_bps, round(close, 2)],
                )

        # factor_panel ------------------------------------------------------
        conn.execute("DROP TABLE IF EXISTS factor_panel")
        conn.execute(
            """
            CREATE TABLE factor_panel (
                factor TEXT,
                as_of_date DATE,
                current_z DOUBLE,
                six_month_avg_z DOUBLE,
                comment TEXT
            )
            """
        )
        for f, z, avg, comment in FACTORS:
            conn.execute(
                "INSERT INTO factor_panel VALUES (?, ?, ?, ?, ?)",
                [f, today, z, avg, comment],
            )

        # economic_indicators ----------------------------------------------
        conn.execute("DROP TABLE IF EXISTS economic_indicators")
        conn.execute(
            """
            CREATE TABLE economic_indicators (
                indicator TEXT,
                date DATE,
                value DOUBLE,
                change_bps INTEGER
            )
            """
        )
        for ind, val, chg in INDICATORS:
            conn.execute(
                "INSERT INTO economic_indicators VALUES (?, ?, ?, ?)",
                [ind, today, val, chg],
            )

        conn.commit()
    finally:
        conn.close()

    typer.echo(f"seeded {db}")


@cli.command()
def query(
    sql: str,
    db: Path = typer.Option(Path("data/duck.db"), "--db"),
) -> None:
    """Quick SQL test on the DuckDB."""
    conn = duckdb.connect(str(db), read_only=True)
    cur = conn.execute(sql)
    cols = [d[0] for d in cur.description] if cur.description else []
    rows = cur.fetchall()
    print(" | ".join(cols))
    print(" | ".join(["---"] * len(cols)))
    for r in rows[:50]:
        print(" | ".join(str(v) for v in r))


if __name__ == "__main__":
    cli()
