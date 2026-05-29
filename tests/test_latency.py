"""Unit tests for the latency harness.

These exercise the timing primitives without spinning up Pipecat. The
real end-to-end harness is the ``alex`` CLI itself, which writes one
JSONL record per turn that ``alex.instrumentation summary`` consumes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from alex.instrumentation import STAGES, TurnLogger, TurnTimer


def test_turn_timer_records_perceived(tmp_path: Path) -> None:
    t = TurnTimer(turn_id="t1")
    t.mark_end_of_speech()
    with t.stage("router"):
        time.sleep(0.005)
    with t.stage("llm_ttft"):
        time.sleep(0.01)
    t.mark_first_audio()

    rec = t.to_record()
    assert rec["turn_id"] == "t1"
    assert rec["durations_ms"]["router"] >= 5
    assert rec["durations_ms"]["llm_ttft"] >= 10
    assert rec["durations_ms"]["perceived"] >= 15


def test_turn_logger_jsonl(tmp_path: Path) -> None:
    log = TurnLogger(tmp_path / "x.jsonl")
    for i in range(3):
        t = TurnTimer(turn_id=f"t{i}")
        t.mark_end_of_speech()
        with t.stage("llm_ttft"):
            pass
        t.mark_first_audio()
        log.write(t)
    log.close()

    lines = (tmp_path / "x.jsonl").read_text().splitlines()
    assert len(lines) == 3
    for line in lines:
        rec = json.loads(line)
        assert "turn_id" in rec
        assert "durations_ms" in rec


def test_known_stages() -> None:
    # Anyone changing STAGES needs to think about backwards-compat of old logs.
    expected = (
        "wake",
        "vad",
        "stt",
        "router",
        "retrieval",
        "llm_start",
        "llm_ttft",
        "llm_done",
        "tts_ttfa",
        "perceived",
    )
    assert STAGES == expected
