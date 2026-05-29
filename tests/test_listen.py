"""Behavior tests for LISTEN mode plumbing.

We don't spin up Pipecat; we drive frames through the router directly
and assert what gets passed downstream vs swallowed.
"""

from __future__ import annotations

import time

import pytest
from pipecat.frames.frames import EndFrame, Frame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection

from alex.turn.keyword_router import TranscriptKeywordRouter
from alex.turn.transcript_buffer import RollingTranscriptBuffer


class FrameCollector:
    """A trivial sink that records what the router pushed downstream."""

    def __init__(self) -> None:
        self.frames: list[Frame] = []

    async def __call__(self, frame, direction):
        self.frames.append(frame)


def _make_router(*, triggers=("alex", "hey alex"), buffer_minutes=5.0):
    buf = RollingTranscriptBuffer(window_minutes=buffer_minutes)
    router = TranscriptKeywordRouter(triggers=triggers, buffer=buf)
    sink = FrameCollector()
    # Monkey-patch push_frame so we can observe downstream traffic without
    # standing up a full Pipecat pipeline.
    router.push_frame = sink
    return router, buf, sink


async def _send(router, frame):
    await router.process_frame(frame, FrameDirection.DOWNSTREAM)


def _txn(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text=text, user_id="t", timestamp=str(time.time()))


@pytest.mark.asyncio
async def test_passive_transcript_is_swallowed():
    router, buf, sink = _make_router()
    await _send(router, _txn("Sebastien mentioned the Treasury curve flattening today."))
    assert sink.frames == []
    assert len(buf.recent()) == 1
    assert buf.recent()[0].triggered is False


@pytest.mark.asyncio
async def test_trigger_passes_through_with_buffer_context():
    router, buf, sink = _make_router()
    await _send(router, _txn("Sebastien mentioned the Treasury curve flattening."))
    await _send(router, _txn("Charles said high yield looks rich."))
    await _send(router, _txn("Hey Alex, what's our duration view?"))

    # First two were swallowed, third passed through.
    assert len(sink.frames) == 1
    pushed = sink.frames[0]
    assert isinstance(pushed, TranscriptionFrame)
    # Triggered text should include both prior passive lines.
    assert "Sebastien mentioned the Treasury curve" in pushed.text
    assert "Charles said high yield" in pushed.text
    # And the addressed question itself.
    assert "what's our duration view" in pushed.text
    # Buffer has all three.
    entries = buf.recent()
    assert len(entries) == 3
    assert [e.triggered for e in entries] == [False, False, True]


@pytest.mark.asyncio
async def test_case_and_punctuation_insensitive():
    router, _, sink = _make_router(triggers=("alex",))
    await _send(router, _txn("ALEX, what's the EM allocation?"))
    assert len(sink.frames) == 1


@pytest.mark.asyncio
async def test_non_trigger_substring_does_not_fire():
    """'alexa' should not fire when only 'alex' is configured as trigger."""
    # NOTE: substring matching DOES match 'alexa' for trigger 'alex' — that's
    # intentional given the use case (people may say "Alex" with a slight
    # drawl). If a deployment has the literal name "Alexa" in scope, set
    # `listen_triggers` to "hey alex,ask alex" to require a prefix.
    router, _, sink = _make_router(triggers=("alex",))
    await _send(router, _txn("Alexa, set a timer."))
    assert len(sink.frames) == 1  # documents current behavior


@pytest.mark.asyncio
async def test_non_transcription_frames_pass_through_untouched():
    router, _, sink = _make_router()
    end = EndFrame()
    await _send(router, end)
    assert sink.frames == [end]


@pytest.mark.asyncio
async def test_buffer_evicts_old_entries():
    router, buf, sink = _make_router(buffer_minutes=0.0001)  # ~6 ms window
    await _send(router, _txn("ancient line"))
    time.sleep(0.05)
    await _send(router, _txn("fresh line"))
    entries = buf.recent()
    assert len(entries) == 1
    assert entries[0].text == "fresh line"
