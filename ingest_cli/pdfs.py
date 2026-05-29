"""Walk ``corpora/`` and load PDFs / markdown / text into the LanceDB store.

Chunking is intentionally simple in v1:
- Markdown: split on top-level (``# ``) and second-level (``## ``) headings,
  keep the heading path as ``section``. Long sections get bucketed.
- PDF: PyMuPDF text extraction page-by-page, then paragraph-split + bucket.
- TXT: paragraph-split + bucket.

Buckets target ~600 chars with up to ~900 (we lean smaller for tight TTS
contexts; the LLM gets multiple chunks). Adjust ``TARGET_CHARS`` / ``MAX_CHARS``
if a domain wants longer cuts.

CLI:
    uv run python -m ingest_cli.pdfs build [--corpus-dir corpora/] [--db data/lance]
    uv run python -m ingest_cli.pdfs query "what's our EM view"
"""

from __future__ import annotations

import re
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from alex.rag.embedder import LocalEmbedder
from alex.rag.lancedb_store import Chunk, LanceCorpus

cli = typer.Typer(no_args_is_help=True, add_completion=False)

TARGET_CHARS = 600
MAX_CHARS = 900
SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


# --- chunking -----------------------------------------------------------------


def _bucket_paragraphs(paragraphs: list[str]) -> list[str]:
    """Greedy paragraph→bucket packing. Each bucket stays under MAX_CHARS;
    we close a bucket as soon as adding the next paragraph would exceed
    TARGET_CHARS and the bucket is already non-empty."""
    buckets: list[list[str]] = []
    cur: list[str] = []
    cur_len = 0
    for p in paragraphs:
        if not p.strip():
            continue
        if cur and cur_len + len(p) > MAX_CHARS:
            buckets.append(cur)
            cur, cur_len = [p], len(p)
            continue
        cur.append(p)
        cur_len += len(p) + 1
        if cur_len >= TARGET_CHARS:
            buckets.append(cur)
            cur, cur_len = [], 0
    if cur:
        buckets.append(cur)
    return ["\n\n".join(b).strip() for b in buckets]


def _chunk_markdown(text: str) -> list[tuple[str, str]]:
    """Return list of (section_path, chunk_text). Splits on H1/H2."""
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    h1 = h2 = ""
    current: list[str] = []

    def _flush():
        if current:
            sections.append(("/".join(p for p in (h1, h2) if p) or "body", current.copy()))
            current.clear()

    for line in lines:
        if line.startswith("# "):
            _flush()
            h1 = line[2:].strip()
            h2 = ""
        elif line.startswith("## "):
            _flush()
            h2 = line[3:].strip()
        else:
            current.append(line)
    _flush()

    out: list[tuple[str, str]] = []
    for section, body_lines in sections:
        body = "\n".join(body_lines).strip()
        if not body:
            continue
        paragraphs = re.split(r"\n\s*\n", body)
        for chunk in _bucket_paragraphs(paragraphs):
            out.append((section, chunk))
    return out


def _chunk_plain(text: str, default_section: str = "body") -> list[tuple[str, str]]:
    paragraphs = re.split(r"\n\s*\n", text.strip())
    return [(default_section, c) for c in _bucket_paragraphs(paragraphs)]


def _read_file(path: Path) -> tuple[str, list[tuple[str, str]]]:
    """Return (title, [(section, chunk_text), ...])."""
    if path.suffix.lower() == ".pdf":
        import pymupdf

        with pymupdf.open(path) as doc:
            text = "\n\n".join(page.get_text() for page in doc)
            title = doc.metadata.get("title") or path.stem
        return title, _chunk_plain(text)
    if path.suffix.lower() == ".md":
        text = path.read_text()
        # First H1, if any, is the title.
        h1 = next((line[2:].strip() for line in text.splitlines() if line.startswith("# ")), None)
        return h1 or path.stem, _chunk_markdown(text)
    text = path.read_text()
    return path.stem, _chunk_plain(text)


# --- CLI ----------------------------------------------------------------------


@cli.command()
def build(
    corpus_dir: Path = typer.Option(
        Path("corpora"), "--corpus-dir", help="Directory to scan for .pdf/.md/.txt."
    ),
    db: Path = typer.Option(
        Path("data/lance"), "--db", help="LanceDB directory (created if missing)."
    ),
    table: str = typer.Option("chunks", "--table"),
    fresh: bool = typer.Option(False, "--fresh", help="Wipe the table before ingesting."),
) -> None:
    """Walk ``corpus_dir`` and ingest every supported file."""
    files = sorted(
        p for p in corpus_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not files:
        typer.echo(f"no supported files under {corpus_dir}")
        raise typer.Exit(0)

    if fresh and db.exists():
        import shutil

        shutil.rmtree(db)
        logger.info(f"wiped {db}")

    embedder = LocalEmbedder()
    corpus = LanceCorpus(db, table=table, dim=embedder.dim)

    all_chunks: list[Chunk] = []
    for path in files:
        title, sections = _read_file(path)
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(path.resolve())))
        for idx, (section, text) in enumerate(sections):
            all_chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    title=title,
                    source=str(path.relative_to(corpus_dir.parent) if path.is_absolute() else path),
                    section=section,
                    chunk_idx=idx,
                    text=text,
                    char_start=0,
                    char_end=len(text),
                )
            )
        logger.info(f"  {path.name}: {len(sections)} chunks")

    if not all_chunks:
        typer.echo("no chunks produced")
        raise typer.Exit(0)

    # Encode in batches; bge-m3 4-bit handles ~32-64 per batch comfortably.
    batch = 32
    import numpy as np

    vecs: list[np.ndarray] = []
    for i in range(0, len(all_chunks), batch):
        chunk_batch = all_chunks[i : i + batch]
        r = embedder.encode([c.text for c in chunk_batch])
        vecs.append(r.vectors)
    vectors = np.concatenate(vecs, axis=0)

    corpus.ingest(all_chunks, vectors)
    corpus.build_fts()

    typer.echo(f"ingested {len(all_chunks)} chunks from {len(files)} files; total rows={corpus.count()}")


@cli.command()
def query(
    text: str,
    db: Path = typer.Option(Path("data/lance"), "--db"),
    table: str = typer.Option("chunks", "--table"),
    k: int = typer.Option(6, "-k", help="Top-k chunks to return"),
) -> None:
    """Run a hybrid (BM25 + dense) search against an ingested corpus."""
    corpus = LanceCorpus(db, table=table)
    if corpus.empty:
        typer.echo("no corpus at that path; run `build` first")
        raise typer.Exit(1)

    embedder = LocalEmbedder()
    qv = embedder.encode([text]).vectors[0]
    hits = corpus.search(query_text=text, query_vector=qv, limit=k)

    table_view = Table(title=f"top {len(hits)} hits for: {text!r}")
    table_view.add_column("score", justify="right", style="cyan")
    table_view.add_column("title")
    table_view.add_column("section")
    table_view.add_column("text", overflow="fold")
    for h in hits:
        table_view.add_row(f"{h.score:.3f}", h.title, h.section, h.text[:220].replace("\n", " "))
    Console().print(table_view)


if __name__ == "__main__":
    cli()
