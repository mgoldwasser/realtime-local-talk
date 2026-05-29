"""Hand-curated entity-relationship lookup.

Sub-millisecond per call. For graph-shaped questions like "which funds hold
both X and Y", "what factors does TLT load on", "what asset class is HYG".

The shape is deliberately tiny — we're not trying to be GraphRAG. Maintain
this by hand as the firm's universe stabilizes; it's documentation that the
LLM can actually consult.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class EntityCorpus:
    data: dict

    @classmethod
    def from_path(cls, path: Path | str) -> Optional["EntityCorpus"]:
        path = Path(path)
        if not path.exists():
            return None
        return cls(data=json.loads(path.read_text()))

    def lookup(self, entity: str) -> dict | None:
        """Return all known facts about ``entity``. Case-insensitive on tickers."""
        e = entity.strip()
        # Exact-match across all top-level categories.
        for cat in ("tickers", "funds", "people", "factor_definitions"):
            bucket = self.data.get(cat, {})
            for key, val in bucket.items():
                if key.lower() == e.lower():
                    return {"category": cat, "key": key, "value": val}
        return None

    def find_funds_holding(self, ticker: str) -> list[str]:
        t = ticker.upper()
        return [
            name for name, info in self.data.get("funds", {}).items()
            if t in (info.get("top_holdings") or [])
        ]

    def find_tickers_by(self, *, asset_class: str | None = None, factor: str | None = None) -> list[str]:
        hits: list[str] = []
        for sym, info in self.data.get("tickers", {}).items():
            if asset_class and info.get("asset_class", "").lower() != asset_class.lower():
                continue
            if factor and factor.lower() not in [f.lower() for f in info.get("factors", [])]:
                continue
            hits.append(sym)
        return hits
