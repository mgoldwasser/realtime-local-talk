"""Hybrid (BM25 + dense) retrieval over LanceDB.

LanceDB is the plan's vector store: embedded, columnar, ARM NEON SIMD tuned
for Apple Silicon. We use the table's native FTS (Tantivy/BM25) plus a dense
vector column, combined via Reciprocal Rank Fusion — both sides of retrieval
without leaving the laptop.

Schema (one row per chunk):
    chunk_id   string  (uuid)
    doc_id     string  (source document id; same across chunks of one doc)
    title      string  (document title — first heading or filename)
    source     string  (file path)
    section    string  (heading path, "/" joined)
    chunk_idx  int
    text       string  (chunk text; indexed for BM25)
    vector     fixed_size_list<float, dim>  (dense embedding)
    char_start int
    char_end   int
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import lancedb
import numpy as np
import pyarrow as pa
from lancedb.rerankers import RRFReranker
from loguru import logger


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    source: str
    section: str
    chunk_idx: int
    text: str
    char_start: int = 0
    char_end: int = 0


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    source: str
    text: str
    score: float
    extras: dict = field(default_factory=dict)


def _arrow_schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            ("chunk_id", pa.string()),
            ("doc_id", pa.string()),
            ("title", pa.string()),
            ("source", pa.string()),
            ("section", pa.string()),
            ("chunk_idx", pa.int32()),
            ("text", pa.string()),
            ("vector", pa.list_(pa.float32(), dim)),
            ("char_start", pa.int32()),
            ("char_end", pa.int32()),
        ]
    )


class LanceCorpus:
    """Owns one LanceDB table (a "corpus") and exposes hybrid search.

    A fresh dataset is created on first ingest. Subsequent ingests append.
    """

    def __init__(
        self,
        db_path: Path | str,
        table: str = "chunks",
        dim: int | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.table_name = table
        self._dim = dim
        self._db = lancedb.connect(str(self.db_path))
        self._tbl = self._open_or_none()

    def _open_or_none(self):
        if self.table_name in self._db.table_names():
            t = self._db.open_table(self.table_name)
            # Infer dim from the existing table for callers that didn't pass one.
            if self._dim is None:
                for field in t.schema:
                    if field.name == "vector":
                        self._dim = field.type.list_size
                        break
            return t
        return None

    @property
    def dim(self) -> int | None:
        return self._dim

    @property
    def empty(self) -> bool:
        return self._tbl is None

    def _ensure_table(self, dim: int) -> None:
        if self._tbl is not None:
            return
        self._dim = dim
        empty = pa.Table.from_pylist([], schema=_arrow_schema(dim))
        self._tbl = self._db.create_table(self.table_name, data=empty, mode="overwrite")
        logger.info(f"created LanceDB table '{self.table_name}' (dim={dim})")

    def ingest(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int:
        if len(chunks) == 0:
            return 0
        if vectors.shape[0] != len(chunks):
            raise ValueError(
                f"vectors/chunks length mismatch: {vectors.shape[0]} vs {len(chunks)}"
            )
        self._ensure_table(vectors.shape[1])

        rows = []
        for c, v in zip(chunks, vectors):
            rows.append(
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "title": c.title,
                    "source": c.source,
                    "section": c.section,
                    "chunk_idx": int(c.chunk_idx),
                    "text": c.text,
                    "vector": v.astype(np.float32).tolist(),
                    "char_start": int(c.char_start),
                    "char_end": int(c.char_end),
                }
            )
        self._tbl.add(rows)
        logger.info(f"ingested {len(rows)} chunks into {self.table_name}")
        return len(rows)

    def build_fts(self) -> None:
        """(Re)build the BM25 index over the ``text`` column."""
        if self._tbl is None:
            return
        self._tbl.create_fts_index("text", replace=True)
        logger.info(f"FTS index built on {self.table_name}.text")

    def search(
        self,
        *,
        query_text: str,
        query_vector: np.ndarray,
        limit: int = 6,
        where: str | None = None,
    ) -> list[Hit]:
        """Hybrid BM25 + dense search, RRF-fused. Returns up to ``limit`` hits."""
        if self._tbl is None:
            return []
        q = (
            self._tbl.search(query_type="hybrid")
            .vector(query_vector.astype(np.float32).tolist())
            .text(query_text)
            .limit(limit)
            .rerank(RRFReranker())
        )
        if where:
            q = q.where(where, prefilter=True)
        rows = q.to_list()

        hits: list[Hit] = []
        for r in rows:
            hits.append(
                Hit(
                    chunk_id=r["chunk_id"],
                    doc_id=r["doc_id"],
                    title=r["title"],
                    section=r.get("section", ""),
                    source=r["source"],
                    text=r["text"],
                    score=float(r.get("_relevance_score", 0.0)),
                    extras={k: r[k] for k in ("chunk_idx", "char_start", "char_end") if k in r},
                )
            )
        return hits

    def count(self) -> int:
        return 0 if self._tbl is None else self._tbl.count_rows()
