"""Behavior tests for the text-based echo filter."""

from __future__ import annotations

import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterruptionFrame,
    LLMTextFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from alex.turn.echo_filter import TextBasedEchoFilter


class _Sink:
    def __init__(self) -> None:
        self.frames: list[Frame] = []

    async def __call__(self, frame, direction):
        self.frames.append(frame)


def _build(overlap_threshold: float = 0.6) -> tuple[TextBasedEchoFilter, _Sink]:
    f = TextBasedEchoFilter(overlap_threshold=overlap_threshold)
    sink = _Sink()
    f.push_frame = sink
    return f, sink


async def _setup_polly_speaking(f, sink, tts_text: str) -> None:
    await f.process_frame(BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await f.process_frame(LLMTextFrame(tts_text), FrameDirection.DOWNSTREAM)
    sink.frames.clear()  # discard lifecycle setup frames


async def _send_transcript(f, sink, text: str) -> tuple[bool, bool]:
    """Send a transcript; return (was_pushed_downstream, was_barge_in)."""
    sink.frames.clear()
    await f.process_frame(
        TranscriptionFrame(text=text, user_id="u", timestamp="t"),
        FrameDirection.DOWNSTREAM,
    )
    pushed = any(
        isinstance(fr, TranscriptionFrame) and fr.text == text for fr in sink.frames
    )
    barge_in = any(isinstance(fr, InterruptionFrame) for fr in sink.frames)
    return pushed, barge_in


@pytest.mark.asyncio
async def test_echo_chunk_suppressed():
    f, sink = _build()
    await _setup_polly_speaking(
        f, sink, "We are modestly overweight EM equities for the first quarter"
    )
    pushed, barge_in = await _send_transcript(f, sink, "modestly overweight EM equities")
    assert not pushed
    assert not barge_in


@pytest.mark.asyncio
async def test_barge_in_fires_interruption():
    f, sink = _build()
    await _setup_polly_speaking(
        f, sink, "We are modestly overweight EM equities for the first quarter"
    )
    pushed, barge_in = await _send_transcript(f, sink, "wait hold on Alex stop")
    assert pushed
    assert barge_in


@pytest.mark.asyncio
async def test_short_transcripts_default_to_suppress():
    """One- or two-token transcripts are too risky to route; treat as echo."""
    f, sink = _build()
    await _setup_polly_speaking(f, sink, "We are talking about emerging markets")
    pushed, barge_in = await _send_transcript(f, sink, "yes")
    assert not pushed
    assert not barge_in


@pytest.mark.asyncio
async def test_transcript_passes_through_when_bot_silent():
    """The filter only suppresses while the bot is speaking."""
    f, sink = _build()
    # No BotStartedSpeakingFrame — bot is silent.
    pushed, barge_in = await _send_transcript(f, sink, "What's our duration view?")
    assert pushed
    assert not barge_in


@pytest.mark.asyncio
async def test_empty_tts_buffer_treats_as_real_user():
    """If TTS hasn't said anything recently, every transcript is real."""
    f, sink = _build()
    await f.process_frame(BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    sink.frames.clear()
    # No LLMTextFrame was sent; the TTS tracker is empty.
    pushed, barge_in = await _send_transcript(f, sink, "Alex what's our duration view")
    # Empty buffer → overlap is 0 → treat as real interrupt
    assert pushed
    assert barge_in


@pytest.mark.asyncio
async def test_stops_filtering_after_bot_finishes():
    f, sink = _build()
    await _setup_polly_speaking(f, sink, "We are modestly overweight EM equities")
    await f.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    sink.frames.clear()
    # Once bot is done, even text matching Polly's earlier output passes through —
    # because we only filter while bot_speaking == True.
    pushed, _ = await _send_transcript(f, sink, "modestly overweight EM equities")
    assert pushed
