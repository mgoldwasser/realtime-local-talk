"""Thin wrapper over Pipecat's ``WhisperSTTServiceMLX`` that exposes
``initial_prompt`` for vocabulary biasing.

Pipecat's upstream service hardcodes the call to ``mlx_whisper.transcribe``
without passing ``initial_prompt`` — which is the canonical Whisper hook for
domain-vocab biasing. We override ``run_stt`` to inject it (plus
``condition_on_previous_text``, which we keep True so multi-utterance turns
benefit from carrying context).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Optional

import numpy as np
from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.settings import assert_given
from pipecat.services.whisper.stt import MLXModel, WhisperSTTServiceMLX
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601


class BiasedMLXWhisperSTTService(WhisperSTTServiceMLX):
    """``WhisperSTTServiceMLX`` + an ``initial_prompt`` for vocab biasing."""

    def __init__(
        self,
        *,
        initial_prompt: Optional[str] = None,
        condition_on_previous_text: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._initial_prompt = initial_prompt
        self._condition_on_previous_text = condition_on_previous_text
        if initial_prompt:
            logger.info(
                f"MLX-Whisper vocab bias active ({len(initial_prompt)} chars)"
            )

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        try:
            import mlx_whisper

            await self.start_processing_metrics()

            audio_float = (
                np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
            )

            model_path = assert_given(self._settings.model)
            temperature = assert_given(self._settings.temperature)
            language = assert_given(self._settings.language)
            no_speech_prob_threshold = assert_given(self._settings.no_speech_prob)

            chunk = await asyncio.to_thread(
                mlx_whisper.transcribe,
                audio_float,
                path_or_hf_repo=model_path,
                temperature=temperature,
                language=language,
                initial_prompt=self._initial_prompt,
                condition_on_previous_text=self._condition_on_previous_text,
            )

            text = ""
            for segment in chunk.get("segments", []):
                # Drop Whisper's known repetition hallucination marker.
                if segment.get("compression_ratio") == 0.5555555555555556:
                    continue
                if segment.get("no_speech_prob", 0.0) < no_speech_prob_threshold:
                    text += f"{segment.get('text', '')} "

            text = text.strip() or None

            await self.stop_processing_metrics()

            if text:
                await self._handle_transcription(text, True, language)
                logger.info(f"📝 heard: {text}")
                yield TranscriptionFrame(
                    text,
                    self._user_id,
                    time_now_iso8601(),
                    language,
                )

        except Exception as e:
            yield ErrorFrame(error=f"MLX-Whisper transcription failed: {e}")


def build_stt_service(
    model: MLXModel = MLXModel.LARGE_V3_TURBO_Q4,
    initial_prompt: Optional[str] = None,
    language: Language = Language.EN,
    no_speech_prob: float = 0.6,
    temperature: float = 0.0,
) -> BiasedMLXWhisperSTTService:
    """Default factory matching the plan's STT choice (Q4 turbo on ANE)."""
    return BiasedMLXWhisperSTTService(
        initial_prompt=initial_prompt,
        settings=BiasedMLXWhisperSTTService.Settings(
            model=model.value,
            language=language,
            no_speech_prob=no_speech_prob,
            temperature=temperature,
        ),
    )
