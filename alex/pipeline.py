"""Phase 1 pipeline: LocalAudioTransport ↔ Bedrock Haiku 4.5 ↔ Polly streaming.

Wake word, STT, RAG, escalation routing arrive in later phases. This module
is intentionally narrow: it proves the audio path works end-to-end and emits
per-stage latency metrics.

Modes:
- ``text_input=True``  → read stdin lines, inject as TranscriptionFrame
  (bypasses mic + STT; useful before Phase 2 lands).
- ``text_input=False`` → mic in, audio out via the local transport.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from datetime import datetime
from typing import Optional

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMRunFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.turns.user_mute import AlwaysUserMuteStrategy
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.aws.llm import AWSBedrockLLMService
from pipecat.services.aws.tts import AWSPollyTTSService
from pipecat.services.whisper.stt import MLXModel
from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)

from alex.config import Activation, LlmTier, Settings
from alex.instrumentation import TurnLogger, TurnTimer
from alex.llm.tools import RagDeps, build_tools
from alex.rag.embedder import LocalEmbedder
from alex.rag.duckdb_store import DuckDBCorpus
from alex.rag.entities import EntityCorpus
from alex.rag.lancedb_store import LanceCorpus
from alex.rag.vocab import load_vocab_prompt
from alex.stt.mlx_whisper_service import build_stt_service
from alex.wake.openwakeword_runner import (
    WakeWordDetector,
    WakeWordMuteStrategy,
    WakeWordState,
)
from alex.wake.ptt import PTTHotkeyListener, PTTMuteStrategy, PTTState


SYSTEM_PROMPT = (
    "You are Alex, a real-time voice meeting assistant for an investment "
    "committee. Speak in short, conversational sentences — you are talking "
    "out loud to a meeting room, not writing prose. Default to one to three "
    "short sentences. If you don't know, say so quickly. Never read out long "
    "URLs, citations, or markdown formatting.\n\n"
    "TOOLS:\n"
    "• `rag_lookup` — call for any question about the firm's positioning, "
    "our prior calls, internal views, factor exposures, TAA committee "
    "decisions, or specific historical research notes. Cite the source by "
    "note title ('per the Q1 EM Equities Outlook…'). If no hits, say so "
    "briefly and answer from general knowledge with a caveat. Do not call "
    "for chitchat, definitions, or general macro context.\n"
    "• `sql_lookup` — call for current numeric / tabular questions like "
    "'what's our current allocation', 'how has SPY traded recently', "
    "'what's the 10-year now'. Generates a SELECT against the structured "
    "warehouse.\n"
    "• `web_search` — call for time-sensitive questions ('today', 'this "
    "morning', 'latest', current prices/news). Do NOT use for evergreen "
    "concepts or anything the other two tools answer.\n"
    "• `escalate_to_sonnet` — call for multi-step finance reasoning, "
    "scenario analysis, comparing trade-offs, or questions where the user "
    "asks 'why' or 'explain' on something non-trivial. Don't escalate "
    "simple lookups."
)


def _model_id_for_tier(settings: Settings, tier: Optional[LlmTier]) -> str:
    tier = tier or LlmTier.HAIKU
    return {
        LlmTier.HAIKU: settings.llm_haiku,
        LlmTier.SONNET: settings.llm_sonnet,
        LlmTier.GPT5: settings.llm_haiku,  # GPT-5 path wired in Phase 5; fall back for now.
    }[tier]


class LatencyTap(FrameProcessor):
    """Drop-in processor that marks turn timings as frames flow through it.

    Pipeline taps:
    - ``endpoint``  ``UserStoppedSpeakingFrame``    end-of-speech anchor (voice)
    - ``stt``       ``TranscriptionFrame``          STT done; opens llm_*
    - ``llm_start`` ``LLMFullResponseStartFrame``   model started responding
    - ``ttft``      ``LLMTextFrame``                first text chunk (post-tools)
    - ``llm_done``  ``LLMFullResponseEndFrame``     full response complete
    - ``ttfa``      ``TTSAudioRawFrame``            first audio chunk
    """

    _MAP = {
        "endpoint": UserStoppedSpeakingFrame,
        "stt": TranscriptionFrame,
        "llm_start": LLMFullResponseStartFrame,
        "ttft": LLMTextFrame,
        "llm_done": LLMFullResponseEndFrame,
        "ttfa": TTSAudioRawFrame,
    }

    def __init__(self, label: str, on_first: callable) -> None:
        super().__init__()
        self._label = label
        self._on_first = on_first
        self._target = self._MAP[label]
        self._fired = False

    def reset(self) -> None:
        self._fired = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if not self._fired and isinstance(frame, self._target):
            self._fired = True
            self._on_first()
        await self.push_frame(frame, direction)


async def run_pipeline(
    settings: Settings,
    text_input: bool,
    forced_tier: Optional[LlmTier],
    silent: bool = False,
) -> None:
    logger.remove()
    logger.add(sys.stderr, level=os.getenv("LOGLEVEL", "INFO"))

    # --- AWS credentials ---------------------------------------------------
    # Both services read boto3's default credential chain; we pass region
    # explicitly so a missing AWS_REGION env still works.
    bedrock_kwargs = {"aws_region": settings.aws_region}
    polly_kwargs = {"region": settings.aws_region}
    if os.getenv("AWS_ACCESS_KEY_ID"):
        bedrock_kwargs.update(
            aws_access_key=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        )

    # --- Services ----------------------------------------------------------
    # latency="optimized" turns on Bedrock latency-optimized inference for
    # Haiku/Llama. enable_prompt_caching keeps the system prompt + persistent
    # context warm across turns — both are in the plan's hot path.
    model_id = _model_id_for_tier(settings, forced_tier)
    logger.info(f"LLM model: {model_id}")
    llm = AWSBedrockLLMService(
        settings=AWSBedrockLLMService.Settings(
            model=model_id,
            system_instruction=SYSTEM_PROMPT,
            max_tokens=400,
            temperature=0.4,
            latency=settings.llm_latency,
            enable_prompt_caching=True,
        ),
        **bedrock_kwargs,
    )

    if silent:
        logger.info("TTS: silent (no audio)")
        tts = None
    else:
        logger.info(f"TTS: Polly {settings.polly_engine} voice={settings.polly_voice}")
        tts = AWSPollyTTSService(
            sample_rate=24000,
            settings=AWSPollyTTSService.Settings(
                voice=settings.polly_voice,
                engine=settings.polly_engine,  # "generative" | "neural" | "long-form"
            ),
            **polly_kwargs,
        )

    # --- Transport ---------------------------------------------------------
    # text_input mode disables the mic and audio output is whatever the
    # system default is. Voice mode wires sounddevice through PortAudio.
    transport = LocalAudioTransport(
        params=LocalAudioTransportParams(
            audio_in_enabled=not text_input,
            audio_out_enabled=True,
            audio_out_sample_rate=24000,
            audio_in_sample_rate=16000,
            vad_analyzer=SileroVADAnalyzer() if not text_input else None,
            input_device_index=_resolve_device(settings.input_device, kind="input"),
            output_device_index=_resolve_device(settings.output_device, kind="output"),
        )
    )

    # --- STT (voice mode only) --------------------------------------------
    stt = None
    if not text_input:
        try:
            model_enum = MLXModel[settings.stt_model]
        except KeyError:
            raise ValueError(
                f"unknown STT model {settings.stt_model!r}; valid: "
                f"{[m.name for m in MLXModel]}"
            )
        vocab_prompt = load_vocab_prompt(settings.vocab_path)
        logger.info(f"STT: MLX-Whisper {model_enum.name}")
        stt = build_stt_service(model=model_enum, initial_prompt=vocab_prompt)

    # --- Speculative filler (Phase 5a) ------------------------------------
    # When the LLM fires a tool call, the real text response is 2-4s away
    # (tool execution + second LLM call). Speak an immediate filler so the
    # turn *feels* responsive. Polly Generative TTFA ~150-300ms means the
    # user hears "Let me check..." while RAG/etc runs in the background.
    import random as _random

    FILLERS = [
        "Let me check that.",
        "Give me a second.",
        "Looking that up now.",
        "One moment, pulling that up.",
        "Let me dig into that.",
        "Checking the research now.",
    ]

    @llm.event_handler("on_function_calls_started")
    async def _on_tool_call(service, calls):  # noqa: ARG001
        if tts is None:
            return
        filler = _random.choice(FILLERS)
        logger.info(f"💬 filler: {filler!r} (tools: {[c.function_name for c in calls]})")
        await tts.queue_frame(TTSSpeakFrame(filler))

    # --- RAG (lazy-loaded; empty corpus is fine) --------------------------
    rag_corpus = LanceCorpus(
        settings.lance_db_path, table=settings.lance_table
    )
    if rag_corpus.empty:
        logger.warning(
            f"no RAG corpus at {settings.lance_db_path} — run "
            f"`uv run python -m ingest_cli.pdfs build` to populate"
        )
        tool_schema = None
    else:
        embedder = LocalEmbedder()
        duck = None
        if settings.duck_db_path.exists():
            duck = DuckDBCorpus(settings.duck_db_path)
            if duck.empty():
                duck.close()
                duck = None
        if duck is None:
            logger.warning(
                f"no DuckDB at {settings.duck_db_path} — sql_lookup tool "
                f"disabled. Run `uv run python -m ingest_cli.datapoints seed` "
                f"to enable structured queries."
            )
        entities = EntityCorpus.from_path(settings.entities_path)
        if entities is None:
            logger.warning(f"no entity corpus at {settings.entities_path}")
        deps = RagDeps(
            embedder=embedder,
            corpus=rag_corpus,
            sonnet_model_id=settings.llm_sonnet,
            aws_region=settings.aws_region,
            duck=duck,
            entities=entities,
        )
        tool_schema, handlers = build_tools(deps)
        for name, fn in handlers.items():
            llm.register_function(name, fn)
        logger.info(
            f"RAG ready: {rag_corpus.count()} chunks, tools={list(handlers)}"
        )

    # --- Context + aggregators --------------------------------------------
    # System prompt lives on the LLM service (Bedrock uses a separate system
    # field in Converse), so we leave LLMContext starting empty.
    context = LLMContext(tools=tool_schema) if tool_schema else LLMContext()
    # Mute strategy selection. Strategies are OR-combined — mute if ANY says
    # to mute. Two orthogonal concerns:
    #   1. The user's activation gate (always-on / PTT / wake) — decides
    #      whether the mic is "open" right now at all.
    #   2. The self-echo guard — must mute *while Alex is speaking* even
    #      inside an open window, otherwise Polly through speakers gets
    #      transcribed as a new user turn and we loop. The exception is
    #      `hardware_aec_present` where the speakerphone cleans the mic.
    ptt_state: Optional[PTTState] = None
    ptt_listener: Optional[PTTHotkeyListener] = None
    wake_state: Optional[WakeWordState] = None
    wake_detector: Optional[WakeWordDetector] = None
    mute_strategies: list = []
    if not text_input:
        # Activation gate.
        if settings.activation in (Activation.PTT, Activation.BOTH):
            ptt_state = PTTState()
            ptt_state.on_toggle = lambda on: logger.info(
                f"🎙  PTT {'ON — listening' if on else 'OFF — muted'}"
            )
            mute_strategies.append(PTTMuteStrategy(ptt_state))
        if settings.activation in (Activation.WAKEWORD, Activation.BOTH):
            wake_state = WakeWordState(
                on_detect=lambda: logger.info(
                    f"🎙  wake — listening for {settings.wakeword_listen_secs:.0f}s"
                )
            )
            wake_detector = WakeWordDetector(
                state=wake_state,
                model_path=settings.wakeword_model_path,
                threshold=settings.wakeword_threshold,
                listen_secs=settings.wakeword_listen_secs,
            )
            mute_strategies.append(WakeWordMuteStrategy(wake_state))

        # Self-echo guard — always on unless hardware AEC handles it.
        if not settings.hardware_aec_present:
            mute_strategies.append(AlwaysUserMuteStrategy())
        else:
            logger.info(
                "hardware AEC mode: software self-mute disabled; barge-in enabled"
            )

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer() if not text_input else None,
            user_mute_strategies=mute_strategies,
        ),
    )

    # --- Latency tracking -------------------------------------------------
    # File prefix labels the active path: voice = voice in / out, text = stdin in.
    prefix = "text" if text_input else "voice"
    log_path = (
        settings.latency_dir
        / f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.jsonl"
    )
    turn_log = TurnLogger(log_path)
    current_timer: dict[str, TurnTimer] = {}  # holds the live timer per turn

    import time as _t

    def _start_turn() -> TurnTimer:
        timer = TurnTimer(turn_id=str(uuid.uuid4())[:8])
        timer.mark_end_of_speech()
        timer._starts["stt"] = _t.perf_counter()
        current_timer["t"] = timer
        for tap in (endpoint_tap, stt_tap, llm_start_tap, ttft_tap, llm_done_tap, ttfa_tap):
            tap.reset()
        return timer

    def _mark_endpoint() -> None:
        # Voice mode: VAD says the user just stopped speaking. This is the
        # canonical end-of-speech anchor for "perceived" latency.
        if "t" not in current_timer:
            _start_turn()

    def _close(t: TurnTimer, stage: str, start_key: str) -> None:
        if stage not in t.durations and start_key in t._starts:
            t.durations[stage] = (_t.perf_counter() - t._starts[start_key]) * 1000

    def _mark_stt() -> None:
        t = current_timer.get("t") or _start_turn()
        _close(t, "stt", "stt")
        # Anchor every downstream LLM stage to STT-done.
        t._starts["llm_anchor"] = _t.perf_counter()

    def _mark_llm_start() -> None:
        # First LLMFullResponseStartFrame wins — that's when the first LLM
        # call (which may be a tool-use decision) actually started streaming.
        t = current_timer.get("t")
        if t and "llm_start" not in t.durations and "llm_anchor" in t._starts:
            t.durations["llm_start"] = (
                _t.perf_counter() - t._starts["llm_anchor"]
            ) * 1000

    def _mark_ttft() -> None:
        t = current_timer.get("t")
        if t and "llm_ttft" not in t.durations and "llm_anchor" in t._starts:
            t.durations["llm_ttft"] = (
                _t.perf_counter() - t._starts["llm_anchor"]
            ) * 1000
            t._starts["tts_ttfa"] = _t.perf_counter()
            # Tool-using turns emit LLMFullResponseEndFrame twice: once after
            # the tool decision, once after the real text streams. The first
            # one already consumed llm_done_tap (single-shot); re-arm it so
            # the second end-frame can close the turn cleanly.
            llm_done_tap.reset()

    def _close_turn() -> None:
        t = current_timer.get("t")
        if not t:
            return
        t.mark_first_audio()  # marks the perceived-end anchor (idempotent)
        turn_log.write(t)
        logger.info(f"⏱  turn {t.turn_id} {t.durations}")
        current_timer.pop("t", None)

    def _mark_llm_done() -> None:
        t = current_timer.get("t")
        if not t:
            return
        # Pipecat fires LLMFullResponseEndFrame once per LLM call. With tool
        # use that means we see it twice: after the tool_use round and again
        # after the real text generation. Wait until we've seen text (ttft)
        # so we close on the truly-final end-frame.
        if "llm_ttft" not in t.durations:
            return
        if "llm_done" not in t.durations and "llm_anchor" in t._starts:
            t.durations["llm_done"] = (
                _t.perf_counter() - t._starts["llm_anchor"]
            ) * 1000
        if silent:
            _close_turn()

    def _mark_ttfa() -> None:
        t = current_timer.get("t")
        if t and "tts_ttfa" not in t.durations and "tts_ttfa" in t._starts:
            t.durations["tts_ttfa"] = (_t.perf_counter() - t._starts["tts_ttfa"]) * 1000
        _close_turn()

    endpoint_tap = LatencyTap("endpoint", _mark_endpoint)
    stt_tap = LatencyTap("stt", _mark_stt)
    llm_start_tap = LatencyTap("llm_start", _mark_llm_start)
    ttft_tap = LatencyTap("ttft", _mark_ttft)
    llm_done_tap = LatencyTap("llm_done", _mark_llm_done)
    ttfa_tap = LatencyTap("ttfa", _mark_ttfa)

    # Surface what Alex is actually saying so we can spot-check grounding
    # without playing the audio back. Buffers tokens; flushes on response end.
    class _SpeechEcho(FrameProcessor):
        def __init__(self) -> None:
            super().__init__()
            self._buf: list[str] = []

        async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
            await super().process_frame(frame, direction)
            if isinstance(frame, LLMTextFrame):
                self._buf.append(frame.text)
            elif self._buf and not isinstance(frame, LLMTextFrame):
                # Flush on any non-text frame after we've buffered some text.
                from pipecat.frames.frames import LLMFullResponseEndFrame

                if isinstance(frame, LLMFullResponseEndFrame):
                    logger.info(f"🗣  said: {''.join(self._buf).strip()}")
                    self._buf.clear()
            await self.push_frame(frame, direction)

    speech_echo = _SpeechEcho()

    # --- Pipeline ----------------------------------------------------------
    # Voice mode: transport.input() emits UserStoppedSpeakingFrame on VAD
    # endpoint → endpoint_tap anchors the timer → STT runs → TranscriptionFrame
    # → stt_tap closes the stt stage and opens llm_ttft → ...
    # Text mode: stdin pushes TranscriptionFrame; endpoint_tap never fires, so
    # stt_tap is the anchor (stt duration ≈ 0).
    stages: list = [transport.input()]
    if wake_detector is not None:
        stages.append(wake_detector)
    stages.append(endpoint_tap)
    if stt is not None:
        stages.append(stt)
    stages.extend(
        [
            stt_tap,
            user_aggregator,
            llm,
            llm_start_tap,
            ttft_tap,
            speech_echo,
            llm_done_tap,
        ]
    )
    if tts is not None:
        stages.extend([tts, ttfa_tap, transport.output()])
    stages.append(assistant_aggregator)
    pipeline = Pipeline(stages)

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
            enable_metrics=True,
            enable_usage_metrics=True,
            report_only_initial_ttfb=False,
        ),
    )

    # --- text_input loop --------------------------------------------------
    # asyncio's add_reader hooks stdin's fd into the event loop so reads
    # don't block a worker thread — that means Ctrl-C cancels cleanly.
    async def _stdin_loop() -> None:
        from datetime import datetime as _dt

        loop = asyncio.get_event_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        def _on_fd_readable() -> None:
            line = sys.stdin.readline()
            queue.put_nowait(line if line else None)

        loop.add_reader(sys.stdin.fileno(), _on_fd_readable)
        logger.info("text-input mode: type a message and press enter. Ctrl-C or Ctrl-D to exit.")
        try:
            while True:
                line = await queue.get()
                if line is None:
                    await task.queue_frame(EndFrame())
                    return
                line = line.strip()
                if not line:
                    continue
                if line.lower() in {"/q", "/quit", "/exit"}:
                    logger.info("exit shortcut received; shutting down")
                    await task.queue_frame(EndFrame())
                    return
                # text mode: stdin → TranscriptionFrame → stt_tap anchors timer.
                await task.queue_frame(
                    TranscriptionFrame(
                        text=line, user_id="cli", timestamp=_dt.utcnow().isoformat()
                    )
                )
                await task.queue_frame(LLMRunFrame())
        finally:
            loop.remove_reader(sys.stdin.fileno())

    runner = PipelineRunner(handle_sigint=True)

    # Start the PTT hotkey listener now that we know the runner is ready.
    if ptt_state is not None:
        ptt_listener = PTTHotkeyListener(ptt_state)
        ptt_listener.start()

    try:
        if text_input:
            # Run both tasks; whichever finishes first triggers cancellation
            # of the other so the process exits cleanly on Ctrl-C or Ctrl-D.
            runner_task = asyncio.create_task(runner.run(task))
            stdin_task = asyncio.create_task(_stdin_loop())
            try:
                _, pending = await asyncio.wait(
                    {runner_task, stdin_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for p in pending:
                    p.cancel()
                # Drain the cancellations so we surface any actual exceptions.
                await asyncio.gather(runner_task, stdin_task, return_exceptions=True)
            except (KeyboardInterrupt, asyncio.CancelledError):
                runner_task.cancel()
                stdin_task.cancel()
        else:
            await runner.run(task)
    finally:
        turn_log.close()
        if ptt_listener is not None:
            ptt_listener.stop()


def _resolve_device(name: str, *, kind: str) -> Optional[int]:
    """Look up a sounddevice index by substring of its name. Empty → default."""
    if not name:
        return None
    try:
        import sounddevice as sd

        for i, dev in enumerate(sd.query_devices()):
            if kind == "input" and dev["max_input_channels"] < 1:
                continue
            if kind == "output" and dev["max_output_channels"] < 1:
                continue
            if name.lower() in dev["name"].lower():
                logger.info(f"resolved {kind} device '{name}' → index {i} ({dev['name']})")
                return i
        logger.warning(f"no {kind} device matched '{name}'; falling back to default")
    except Exception as e:
        logger.warning(f"device lookup failed: {e}")
    return None
