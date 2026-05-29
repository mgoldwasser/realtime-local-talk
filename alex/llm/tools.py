"""LLM-callable tools exposed to Bedrock Claude in voice mode.

Phase 4 ships ``rag_lookup``. Phases 5/6 will add ``sql_lookup``,
``web_search``, and a small entity-relationship lookup.

Each tool exposes (a) a Pipecat ``FunctionSchema`` so the model can pick
it, and (b) an async handler that takes ``FunctionCallParams`` and returns
JSON via ``params.result_callback``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams

from alex.llm.escalation import ESCALATE_SCHEMA, make_escalate_handler
from alex.llm.web_search import WEB_SEARCH_SCHEMA, make_web_search
from alex.rag.duckdb_store import DB_SCHEMA_DESCRIPTION, DuckDBCorpus
from alex.rag.embedder import LocalEmbedder
from alex.rag.entities import EntityCorpus
from alex.rag.lancedb_store import LanceCorpus


# --- rag_lookup ---------------------------------------------------------------


RAG_LOOKUP_SCHEMA = FunctionSchema(
    name="rag_lookup",
    description=(
        "Search the firm's internal research notes (TAA committee memos, "
        "macro reviews, factor exposure analyses, FOMC reviews). Use this "
        "for any question about positioning, prior calls, internal views, "
        "or specific historical decisions. Returns up to 6 short excerpts "
        "with title and section. Cite the title in your spoken answer."
    ),
    properties={
        "query": {
            "type": "string",
            "description": (
                "Natural-language search query. Keep it focused on the "
                "specific topic, not the user's full question."
            ),
        },
        "limit": {
            "type": "integer",
            "description": "How many chunks to return (default 6, max 10).",
            "minimum": 1,
            "maximum": 10,
        },
    },
    required=["query"],
)


@dataclass
class RagDeps:
    """Heavy objects the rag handler closes over. Built once at pipeline start."""

    embedder: LocalEmbedder
    corpus: LanceCorpus
    sonnet_model_id: str = ""
    aws_region: str = "us-east-1"
    duck: Optional[DuckDBCorpus] = None
    entities: Optional[EntityCorpus] = None
    bus: Any = None  # UIEventBus | None — web dashboard event sink


def make_rag_lookup(deps: RagDeps):
    """Return a Pipecat function handler bound to a corpus + embedder."""

    async def rag_lookup(params: FunctionCallParams) -> None:
        query = (params.arguments.get("query") or "").strip()
        limit = int(params.arguments.get("limit", 6))
        limit = max(1, min(limit, 10))

        if not query:
            await params.result_callback({"error": "query is required"})
            return

        if deps.corpus.empty:
            await params.result_callback(
                {
                    "hits": [],
                    "note": (
                        "No corpus ingested yet. Tell the user that the research "
                        "library hasn't been loaded and answer from general knowledge."
                    ),
                }
            )
            return

        qv = deps.embedder.encode([query]).vectors[0]
        hits = deps.corpus.search(query_text=query, query_vector=qv, limit=limit)
        logger.info(f"🔎 rag_lookup({query!r}, limit={limit}) → {len(hits)} hits")
        if deps.bus:
            deps.bus.emit("state", state="searching")
            deps.bus.emit("tool", name="rag_lookup", detail=query)
            for h in hits[:3]:
                deps.bus.emit(
                    "citation",
                    title=h.title,
                    section=h.section,
                    excerpt=h.text[:280],
                    score=round(h.score, 3),
                )

        result: dict[str, Any] = {
            "query": query,
            "hits": [
                {
                    "title": h.title,
                    "section": h.section,
                    "excerpt": h.text[:500],
                    "score": round(h.score, 4),
                }
                for h in hits
            ],
        }
        await params.result_callback(result)

    return rag_lookup


# --- toolkit builder ----------------------------------------------------------


SQL_LOOKUP_SCHEMA = FunctionSchema(
    name="sql_lookup",
    description=(
        "Query the structured TAA data warehouse (current allocations, "
        "time-series returns, factor exposures, economic indicators) via "
        "read-only DuckDB SQL. Use this for numeric / tabular questions "
        "the research notes won't answer — current allocations, recent "
        "ticker performance, factor z-scores, economic levels. Always "
        "include a LIMIT clause unless the question is single-row.\n\n"
        f"{DB_SCHEMA_DESCRIPTION}"
    ),
    properties={
        "sql": {
            "type": "string",
            "description": "A read-only SELECT statement using the schema above.",
        },
    },
    required=["sql"],
)


def make_sql_lookup(deps: RagDeps):
    async def sql_lookup(params: FunctionCallParams) -> None:
        sql = (params.arguments.get("sql") or "").strip()
        if not sql:
            await params.result_callback({"error": "sql is required"})
            return
        if deps.duck is None:
            await params.result_callback(
                {"error": "structured database not loaded; run ingest_cli.datapoints seed"}
            )
            return
        logger.info(f"🛢  sql_lookup: {sql[:140]}")
        if deps.bus:
            deps.bus.emit("state", state="searching")
            deps.bus.emit("tool", name="sql_lookup", detail=sql[:200])
        result = deps.duck.query(sql, row_limit=200)
        await params.result_callback(
            {"sql": sql, "result": result.to_text(max_rows=20), "row_count": len(result.rows)}
        )

    return sql_lookup


ENTITY_LOOKUP_SCHEMA = FunctionSchema(
    name="entity_lookup",
    description=(
        "Look up firm-level relationships: ticker → asset class / region / "
        "sector / factor exposures, fund → top holdings, person → role. "
        "Call for graph-style questions like 'what asset class is HYG', "
        "'what factors does TLT load on', 'which T. Rowe Price funds hold "
        "MSFT', 'who is Sebastien Page'. Sub-millisecond — no penalty for "
        "calling speculatively."
    ),
    properties={
        "entity": {
            "type": "string",
            "description": (
                "Ticker, fund name, person, or factor to look up. Optionally "
                "prefix with the lookup mode: 'funds_holding:MSFT' to find "
                "funds containing a ticker, or 'tickers_in:EM Equity' to "
                "find tickers in an asset class."
            ),
        },
    },
    required=["entity"],
)


def make_entity_lookup(deps: RagDeps):
    async def entity_lookup(params: FunctionCallParams) -> None:
        if deps.entities is None:
            await params.result_callback({"error": "entity corpus not loaded"})
            return
        raw = (params.arguments.get("entity") or "").strip()
        if not raw:
            await params.result_callback({"error": "entity is required"})
            return
        if deps.bus:
            deps.bus.emit("state", state="searching")
            deps.bus.emit("tool", name="entity_lookup", detail=raw)

        # Allow simple "mode:value" prefixes for inverted lookups.
        if ":" in raw:
            mode, _, val = raw.partition(":")
            mode = mode.strip().lower()
            val = val.strip()
            if mode in {"funds_holding", "holds"}:
                hits = deps.entities.find_funds_holding(val)
                await params.result_callback({"funds_holding": val, "results": hits})
                return
            if mode in {"tickers_in", "asset_class"}:
                hits = deps.entities.find_tickers_by(asset_class=val)
                await params.result_callback({"asset_class": val, "tickers": hits})
                return
            if mode in {"factor", "factor_exposed"}:
                hits = deps.entities.find_tickers_by(factor=val)
                await params.result_callback({"factor": val, "tickers": hits})
                return

        result = deps.entities.lookup(raw)
        if result is None:
            await params.result_callback({"entity": raw, "found": False})
            return
        await params.result_callback({"entity": raw, "found": True, **result})

    return entity_lookup


def build_tools(deps: RagDeps) -> tuple[ToolsSchema, dict]:
    """Compose the tool schema + a name→handler map."""
    handlers: dict = {"rag_lookup": make_rag_lookup(deps)}
    tools = [RAG_LOOKUP_SCHEMA]
    if deps.duck is not None:
        handlers["sql_lookup"] = make_sql_lookup(deps)
        tools.append(SQL_LOOKUP_SCHEMA)
    if deps.entities is not None:
        handlers["entity_lookup"] = make_entity_lookup(deps)
        tools.append(ENTITY_LOOKUP_SCHEMA)
    if deps.sonnet_model_id:
        handlers["escalate_to_sonnet"] = make_escalate_handler(
            deps.sonnet_model_id, deps.aws_region
        )
        tools.append(ESCALATE_SCHEMA)
    web = make_web_search()
    if web is not None:
        handlers["web_search"] = web
        tools.append(WEB_SEARCH_SCHEMA)
    schema = ToolsSchema(standard_tools=tools)
    return schema, handlers
