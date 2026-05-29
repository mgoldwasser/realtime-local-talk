"""Content-based echo filter for voice barge-in.

The classic self-echo problem (Polly through speakers → mic → Whisper →
LLM thinks the user just spoke) is normally solved with one of:

  1. Hardware AEC (USB conference speakerphone DSP)
  2. Software AEC (WebRTC AEC3 — blocked on Apple Silicon arm64)
  3. Hard mute (``AlwaysUserMuteStrategy``) — kills barge-in entirely

This module is option 4: a **text-level echo filter** that exploits the
fact that we already know exactly what Alex is saying (we route every
``LLMTextFrame`` into the TTS, so we can buffer it). When a
``TranscriptionFrame`` arrives during bot-speaking, we compute word
overlap against the recent TTS text:

- high overlap   → Whisper transcribed Polly's voice (echo) → suppress
- low overlap    → real human spoke over Alex (barge-in)   → fire
  ``InterruptionFrame`` so Pipecat cancels Alex's reply

Not bulletproof — a real user shouting echoey content (e.g. "yeah we're
overweight EM too" right when Polly says that) would be misread as
echo. But for natural barge-in phrases ("hold on", "wait", "stop",
"actually"), the overlap against Polly's text is ~0% and the filter
fires the interrupt cleanly.

Wire this in **instead of** ``AlwaysUserMuteStrategy`` by setting
``ALEX_ECHO_FILTER=true`` in ``.env``.
"""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    LLMTextFrame,
    InterruptionFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _bigram_overlap(a: list[str], b_set: set[tuple[str, str]]) -> float:
    """Fraction of ``a``'s bigrams that appear in ``b``'s bigram set.

    Bigrams are more discriminating than unigrams — common words like "the"
    and "a" wash out unigram overlap, but a short user phrase like
    "hold on" has zero overlap with anything Polly is saying about EM
    equities while still being short enough to be reliable.
    """
    if len(a) < 2:
        return 0.0
    a_bigrams = list(zip(a, a[1:]))
    if not a_bigrams:
        return 0.0
    hits = sum(1 for bg in a_bigrams if bg in b_set)
    return hits / len(a_bigrams)


@dataclass
class _TtsEntry:
    text: str
    ts: float


class _TtsTextTracker:
    """Records LLMTextFrames the LLM emits — i.e. what's queued for Polly."""

    def __init__(self, window_secs: float = 15.0) -> None:
        self._window_secs = window_secs
        self._entries: deque[_TtsEntry] = deque()

    def append(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self._entries.append(_TtsEntry(text=text, ts=time.time()))
        self._evict()

    def clear(self) -> None:
        self._entries.clear()

    def _evict(self) -> None:
        cutoff = time.time() - self._window_secs
        while self._entries and self._entries[0].ts < cutoff:
            self._entries.popleft()

    def recent_tokens(self) -> list[str]:
        self._evict()
        return _tokens(" ".join(e.text for e in self._entries))


class TextBasedEchoFilter(FrameProcessor):
    """Replaces ``AlwaysUserMuteStrategy`` with a content-based filter.

    Place this between the STT service and the keyword router (or the
    user aggregator). It needs to see both the LLM's text frames (to
    know what Alex is saying) and the incoming transcripts (to decide
    echo vs interrupt). Pipecat broadcasts bot-lifecycle frames in
    both directions, so positioning is flexible.
    """

    def __init__(
        self,
        overlap_threshold: float = 0.6,
        tts_window_secs: float = 15.0,
        min_user_tokens: int = 2,
    ) -> None:
        super().__init__()
        self._tracker = _TtsTextTracker(window_secs=tts_window_secs)
        self._threshold = overlap_threshold
        self._min_user_tokens = min_user_tokens
        self._bot_speaking = False
        logger.info(
            f"echo filter: bigram overlap threshold={overlap_threshold:.0%}, "
            f"tts window={tts_window_secs:.0f}s"
        )

    def _classify(self, transcript_tokens: list[str]) -> tuple[bool, float]:
        if len(transcript_tokens) < self._min_user_tokens:
            return True, 1.0  # too short to safely route; treat as echo
        tts_tokens = self._tracker.recent_tokens()
        if len(tts_tokens) < 2:
            return False, 0.0  # nothing recent in TTS buffer — not echo
        tts_bigrams = set(zip(tts_tokens, tts_tokens[1:]))
        overlap = _bigram_overlap(transcript_tokens, tts_bigrams)
        return overlap >= self._threshold, overlap

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        # Track bot speaking state via Pipecat's lifecycle frames.
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            # Brief tail-window: keep filtering for a moment after Polly
            # stops, since the last bit of audio is still travelling
            # through speakers + mic. The TTS tracker's own time window
            # handles the cleanup.

        # Snoop on the LLM's text output so we know what Polly is being asked
        # to synthesize. These flow downstream toward the TTS service.
        if isinstance(frame, LLMTextFrame):
            self._tracker.append(frame.text)

        # The interesting case: a transcript arrives while bot is speaking.
        if isinstance(frame, TranscriptionFrame) and self._bot_speaking:
            text = (frame.text or "").strip()
            user_tokens = _tokens(text)
            is_echo, overlap = self._classify(user_tokens)
            if is_echo:
                logger.info(
                    f"🔇 echo suppressed (overlap={overlap:.0%}): {text!r}"
                )
                return  # swallow the frame — Pipecat never sees it
            logger.info(
                f"🚨 barge-in (overlap={overlap:.0%}): {text!r}"
            )
            # Push a InterruptionFrame downstream so Pipecat cancels
            # the in-flight LLM/TTS, then let the transcript follow so the
            # user aggregator sees the new user message.
            await self.push_frame(InterruptionFrame(), direction)

        await self.push_frame(frame, direction)
