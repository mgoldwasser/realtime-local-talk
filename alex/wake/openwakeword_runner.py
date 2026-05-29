"""openWakeWord runtime integration.

Two pieces:

1. ``WakeWordDetector`` — a Pipecat ``FrameProcessor`` that snoops on
   incoming ``InputAudioRawFrame`` audio, runs openWakeWord inference,
   and flips a shared ``WakeWordState`` when the trigger phrase is
   detected. Sleeps for ``listen_secs`` after each detection so the
   user has a window to actually ask their question.

2. ``WakeWordMuteStrategy`` — a Pipecat user-mute strategy that reads
   the shared state and mutes everything outside the listen window.

The default model is the bundled ``alexa_v0.1.onnx`` (openWakeWord's
pre-trained Amazon Alexa detector) which fires on "Alex" / "Alexa"
reasonably well as a placeholder. Train a true "Hey Alex" model with
``alex/wake/trainer/`` and point ``settings.wakeword_model_path`` at
the resulting ``.onnx``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from loguru import logger
from pipecat.frames.frames import Frame, InputAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.turns.user_mute import BaseUserMuteStrategy


@dataclass
class WakeWordState:
    """Shared state — the listen window the detector opens and the strategy reads."""

    listen_until: float = 0.0           # unix timestamp; listen if now < listen_until
    on_detect: Optional[Callable[[], None]] = None

    def is_listening(self) -> bool:
        return time.time() < self.listen_until

    def open(self, listen_secs: float) -> None:
        self.listen_until = time.time() + listen_secs
        if self.on_detect:
            self.on_detect()


class WakeWordMuteStrategy(BaseUserMuteStrategy):
    """Mutes input outside the wake-word's listen window."""

    def __init__(self, state: WakeWordState) -> None:
        super().__init__()
        self._state = state

    async def process_frame(self, frame: Frame) -> bool:  # noqa: ARG002
        return not self._state.is_listening()


class WakeWordDetector(FrameProcessor):
    """Runs openWakeWord on incoming mic audio. Place right after transport.input()."""

    def __init__(
        self,
        state: WakeWordState,
        model_path: Optional[Path | str] = None,
        threshold: float = 0.5,
        listen_secs: float = 10.0,
        chunk_samples: int = 1280,   # 80 ms @ 16 kHz, openWakeWord's expected chunk
        log_min_score: float = 0.05,  # log any score above this so you can tune
    ) -> None:
        super().__init__()
        self._state = state
        self._threshold = threshold
        self._listen_secs = listen_secs
        self._chunk = chunk_samples
        self._buf = bytearray()
        self._log_min_score = log_min_score
        self._high_water_score = 0.0
        self._frames_seen = 0

        from openwakeword.model import Model

        # Default to the bundled `alexa_v0.1.onnx` (phonetically close to
        # "Alex"). Without an explicit path, openWakeWord would load every
        # bundled model — incl. hey_jarvis / weather / timer — which would
        # all fire false positives during meetings.
        if model_path is None:
            import openwakeword as oww
            model_path = (
                Path(oww.__file__).parent / "resources" / "models" / "alexa_v0.1.onnx"
            )
        model_path = Path(model_path)

        self._model = Model(wakeword_model_paths=[str(model_path)])
        self._target_model = model_path.stem
        logger.info(
            f"openWakeWord detector ready (threshold={threshold}, "
            f"listen_window={listen_secs:.0f}s, model={self._target_model})"
        )

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        # We're a passive detector — never swallow frames, always pass through.
        if isinstance(frame, InputAudioRawFrame):
            self._frames_seen += 1
            self._buf.extend(frame.audio)
            # Run inference on each full chunk we've buffered.
            need = self._chunk * 2  # int16 = 2 bytes per sample
            while len(self._buf) >= need:
                chunk = bytes(self._buf[:need])
                del self._buf[:need]
                samples = np.frombuffer(chunk, dtype=np.int16)
                preds = self._model.predict(samples)
                self._evaluate(preds)
        await self.push_frame(frame, direction)

    def _evaluate(self, preds: dict) -> None:
        # Pick the score we care about.
        if self._target_model and self._target_model in preds:
            score = float(preds[self._target_model])
            name = self._target_model
        else:
            name, score = max(
                ((n, float(s)) for n, s in preds.items()),
                key=lambda x: x[1],
                default=(None, 0.0),
            )
            if name is None:
                return

        # Tuning log: any score above the noise floor surfaces so you can
        # see what your voice + room actually produces.
        if score > self._high_water_score:
            self._high_water_score = score
        if score >= self._log_min_score:
            logger.info(
                f"wake score={score:.3f} (model={name}, threshold={self._threshold}, "
                f"high_water={self._high_water_score:.3f})"
            )

        if score >= self._threshold:
            logger.info(f"🎯 wake word ({name}) fired score={score:.2f}")
            self._state.open(self._listen_secs)
            self._high_water_score = 0.0  # reset so next utterance shows fresh peak
