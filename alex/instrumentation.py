"""Per-turn latency instrumentation.

Records start/finish timestamps for each pipeline stage, then writes one JSONL
record per turn to ``latency_runs/<session>.jsonl``. The harness in
``tests/test_latency.py`` and the ``summary`` CLI both consume that format.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

# Stages we time. Order matters for the printed table; missing stages render as "-".
STAGES = (
    "wake",          # wake-word detection (or PTT press → release)
    "vad",           # VAD endpoint + semantic turn detection
    "stt",           # final STT flush
    "router",        # local classifier
    "retrieval",     # RAG: vector + SQL + cache lookup
    "llm_start",     # STT-done → LLM started responding (first model frame)
    "llm_ttft",      # STT-done → first text frame (post-tools if any)
    "llm_done",      # STT-done → LLM finished responding
    "tts_ttfa",      # first text → first TTS audio
    "perceived",     # end-of-speech → first audio audible to user
)


@dataclass
class TurnTimer:
    """Collect per-stage durations for one user turn. Use ``stage(name)`` as a
    context manager; the elapsed ms is stored on ``durations``."""

    turn_id: str
    durations: dict[str, float] = field(default_factory=dict)
    _starts: dict[str, float] = field(default_factory=dict)
    _t0: Optional[float] = None  # end-of-speech anchor for "perceived"

    def mark_end_of_speech(self) -> None:
        self._t0 = time.perf_counter()

    def mark_first_audio(self) -> None:
        if self._t0 is not None:
            self.durations["perceived"] = (time.perf_counter() - self._t0) * 1000

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.durations[name] = (time.perf_counter() - start) * 1000

    def to_record(self) -> dict:
        return {"turn_id": self.turn_id, "durations_ms": self.durations}


class TurnLogger:
    """Append turn records to a JSONL file, one per session."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fh = open(path, "a", buffering=1)  # line-buffered

    def write(self, timer: TurnTimer) -> None:
        self._fh.write(json.dumps(timer.to_record()) + "\n")

    def close(self) -> None:
        self._fh.close()


# --- CLI: print a summary table over a JSONL file -----------------------------


def _summary(path: Path) -> None:
    """Print p50/p95 per stage over a latency JSONL file."""
    import statistics

    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        typer.echo("no records")
        raise typer.Exit(1)

    table = Table(title=f"Latency summary: {path.name} (n={len(records)})")
    table.add_column("stage", style="cyan")
    table.add_column("count", justify="right")
    table.add_column("p50 ms", justify="right")
    table.add_column("p95 ms", justify="right")
    table.add_column("max ms", justify="right")

    for stage in STAGES:
        vals = [r["durations_ms"][stage] for r in records if stage in r["durations_ms"]]
        if not vals:
            table.add_row(stage, "0", "-", "-", "-")
            continue
        p50 = statistics.median(vals)
        p95 = statistics.quantiles(vals, n=20)[-1] if len(vals) >= 2 else vals[0]
        table.add_row(stage, str(len(vals)), f"{p50:.0f}", f"{p95:.0f}", f"{max(vals):.0f}")

    Console().print(table)


if __name__ == "__main__":
    typer.run(_summary)
