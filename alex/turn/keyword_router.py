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
from typing import Sequence

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    LLMRunFrame,
    StartFrame,
    TranscriptionFrame,
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
        logger.info(f"keyword router: triggers={list(self._triggers)}")

    def _is_trigger(self, text: str) -> str | None:
        norm = _normalize(text)
        for trig in self._triggers:
            if trig in norm:
                return trig
        return None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        # Pipecat lifecycle frames always pass through (StartFrame, EndFrame,
        # CancelFrame, …); we only intercept transcripts and the run-trigger.
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
            self._buffer.add(text, triggered=matched is not None)

            if matched is None:
                # Passive room talk — log it, but keep the LLM silent.
                logger.info(f"👂 passive: {text}")
                self._last_was_suppressed = True
                return

            self._last_was_suppressed = False
            logger.info(f"⚡ triggered ({matched!r}): {text}")
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
