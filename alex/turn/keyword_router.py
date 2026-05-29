"""Gate the LLM on trigger-phrase presence in the user's transcribed turn.

Every ``TranscriptionFrame`` flowing through this processor is logged to
the rolling buffer. If the text contains one of the configured trigger
phrases, the frame is *rewritten* to include the recent buffer as
context, then passed downstream so the rest of the pipeline runs as
usual. If it doesn't contain a trigger, the frame is dropped — Pipecat's
context aggregator never sees it, and the LLM stays silent.

This lets Alex listen passively for ~30 minutes of meeting talk and
respond only when explicitly addressed, while still having the full
recent conversation available as context on each response.
"""

from __future__ import annotations

import re
import time
from typing import Sequence

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    LLMRunFrame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from alex.turn.transcript_buffer import RollingTranscriptBuffer


def _normalize(s: str) -> str:
    """Lowercase + strip punctuation for forgiving trigger matching."""
    return re.sub(r"[^\w\s]", " ", s.lower())


class TranscriptKeywordRouter(FrameProcessor):
    def __init__(
        self,
        triggers: Sequence[str],
        buffer: RollingTranscriptBuffer,
        suppress_llm_run: bool = True,
        follow_up_initial_secs: float = 10.0,
        follow_up_extend_secs: float = 6.0,
    ) -> None:
        super().__init__()
        # Pre-normalize triggers for fast substring match.
        self._triggers = tuple(_normalize(t) for t in triggers if t.strip())
        self._buffer = buffer
        self._suppress_llm_run = suppress_llm_run
        # Tracks whether the most recent transcript was swallowed, so we can
        # also swallow the trailing LLMRunFrame that the text-input loop pushes
        # alongside every transcript. Without this, Bedrock receives an empty
        # context and throws "A conversation must start with a user message".
        self._last_was_suppressed = False
        # Follow-up window — open after Alex finishes so a natural back-and-
        # forth doesn't require re-saying the trigger phrase. Two knobs:
        # initial_secs is the grace period right after Alex stops; each
        # subsequent UserStartedSpeakingFrame pushes the deadline forward by
        # extend_secs so the window stays open while someone is actively
        # responding. Silence past the deadline = conversation moved on.
        self._follow_up_initial_secs = follow_up_initial_secs
        self._follow_up_extend_secs = follow_up_extend_secs
        self._follow_up_until = 0.0
        logger.info(
            f"keyword router: triggers={list(self._triggers)}, "
            f"follow_up_initial={follow_up_initial_secs:.0f}s, "
            f"follow_up_extend={follow_up_extend_secs:.0f}s"
        )

    def _in_follow_up(self) -> bool:
        return time.time() < self._follow_up_until

    def _extend_follow_up(self, secs: float) -> None:
        """Push the follow-up deadline forward (never shorten it)."""
        new_until = time.time() + secs
        if new_until > self._follow_up_until:
            self._follow_up_until = new_until

    def _is_trigger(self, text: str) -> str | None:
        norm = _normalize(text)
        for trig in self._triggers:
            if trig in norm:
                return trig
        return None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        # Bot speech lifecycle + VAD jointly drive the follow-up window:
        # - Bot stops    → seed the deadline with the initial grace
        # - User starts  → extend the deadline (someone is actively replying)
        # - Bot starts   → close the window (Alex is talking again)
        if isinstance(frame, BotStoppedSpeakingFrame):
            self._extend_follow_up(self._follow_up_initial_secs)
            logger.info(
                f"🎤 follow-up window: {self._follow_up_initial_secs:.0f}s "
                f"grace, +{self._follow_up_extend_secs:.0f}s per utterance"
            )
        elif isinstance(frame, UserStartedSpeakingFrame) and self._in_follow_up():
            # User is speaking inside the window — extend so a pause-to-think
            # mid-reply doesn't kill the window before they finish.
            self._extend_follow_up(self._follow_up_extend_secs)
        elif isinstance(frame, BotStartedSpeakingFrame) and self._follow_up_until:
            self._follow_up_until = 0.0

        # Swallow LLMRunFrames that the text-input loop emits after a
        # suppressed transcript. In voice mode the user aggregator fires the
        # run frame internally only when it actually has a user message to
        # commit, so this guard is a no-op there.
        if isinstance(frame, LLMRunFrame) and self._last_was_suppressed:
            self._last_was_suppressed = False
            return

        if isinstance(frame, TranscriptionFrame):
            text = (frame.text or "").strip()
            if not text:
                await self.push_frame(frame, direction)
                return

            matched = self._is_trigger(text)
            in_follow_up = self._in_follow_up()
            self._buffer.add(text, triggered=(matched is not None) or in_follow_up)

            if matched is None and not in_follow_up:
                # Passive room talk — log it, but keep the LLM silent.
                logger.info(f"👂 passive: {text}")
                self._last_was_suppressed = True
                return

            self._last_was_suppressed = False
            if matched is not None:
                logger.info(f"⚡ triggered ({matched!r}): {text}")
            else:
                logger.info(f"💬 follow-up: {text}")
            # Don't close the window here — VAD will keep extending it while
            # the user is speaking, and BotStartedSpeakingFrame closes it
            # cleanly when Alex begins replying.
            # Rewrite the frame so the LLM sees recent room context first,
            # then the addressed question. We do not modify the buffer entry
            # itself — only the text handed to the LLM aggregator.
            context_dump = self._buffer.render_for_llm(max_chars=4000)
            if context_dump:
                composed = (
                    "Recent room conversation (most-recent last; '→' marks "
                    "turns you replied to):\n"
                    f"{context_dump}\n\n"
                    f"You were just addressed: \"{text}\""
                )
            else:
                composed = text

            await self.push_frame(
                TranscriptionFrame(
                    text=composed,
                    user_id=frame.user_id,
                    timestamp=frame.timestamp,
                    language=frame.language,
                ),
                direction,
            )
            return

        # LLMRunFrame in upstream code (text-input loop pushes one) should
        # only flow through when we just triggered. The router doesn't see
        # those in LISTEN voice mode — Pipecat aggregators fire the LLM run
        # implicitly via the smart-turn detector after our rewritten frame.
        await self.push_frame(frame, direction)
