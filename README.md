# Alex — Real-Time Voice Meeting Assistant

A low-latency voice assistant that sits in a conference room (or on your desk), continuously transcribes the conversation, and responds intelligently — pulling from your firm's research notes, structured data, the live web, and prior turns of the same meeting — when someone addresses it by name.

Built for use cases like a Tactical Asset Allocation committee: investment commentary, pre-meeting research, hundreds of research notes plus thousands of structured data points, all on hand and available sub-second.

Runs on a MacBook Pro M3/M4 with 48 GB unified memory. Uses **AWS Bedrock** (Claude Haiku 4.5 + Sonnet 4.5, Polly, optionally Transcribe and Nova Sonic) and **OpenAI** (Responses API for web search) as the only cloud dependencies; everything else is local (MLX-Whisper for STT, bge-m3 for embeddings, LanceDB + DuckDB for retrieval).

## Quick start

```bash
# 1. System deps (one-time)
brew install portaudio swig

# 2. Python deps
uv sync

# 3. Credentials
cp .env.example .env
# Edit .env: set AWS_REGION, AWS_PROFILE (or keys), optionally OPENAI_API_KEY

# 4. Ingest the sample corpus
uv run python -m ingest_cli.pdfs build --fresh
uv run python -m ingest_cli.datapoints seed --fresh

# 5. Run
uv run alex --mode listen
```

Then start talking. Anything anyone says gets transcribed (`👂 passive: …`). The first time someone says "Alex" (or "Hey Alex" or "Ask Alex"), Alex responds with the prior 5 minutes of conversation as context. Subsequent follow-up replies don't need the trigger phrase — the VAD-driven follow-up window keeps the mic open while the conversation flows.

## What it does

- **Listens continuously** without recording uploads — MLX-Whisper transcribes every utterance locally.
- **Knows when to speak** — substring match for trigger phrases ("alex"), with a VAD-driven follow-up window so natural back-and-forth doesn't need re-addressing.
- **Knows where to look** — Claude Haiku decides per turn whether the answer is in the research notes (`rag_lookup`), the structured warehouse (`sql_lookup`), the entity graph (`entity_lookup`), the live web (`web_search`), or needs a deeper reasoning pass (`escalate_to_sonnet`).
- **Cites sources** — quotes the note title in spoken answers ("per the Q1 EM Equities Outlook…").
- **Hides latency** — speculative filler ("Let me check that…") plays the moment a tool call fires, masking the round-trip.
- **Stays on the laptop** — sensitive research never leaves the Mac; only the query + retrieved snippets touch the cloud LLM.

## Activation modes

| Mode | What it does | When to use |
|---|---|---|
| **`listen`** (recommended) | Continuous transcription; LLM fires on trigger phrase or inside follow-up window | Meeting-assistant use case |
| `wakeword` | Bundled openWakeWord ONNX (fires on "Alexa") gates the mic | Battery-constrained / always-off deployments |
| `ptt` | ⌥-Space hotkey toggles the mic | Desk use, when you want full control |
| `both` | Wake word OR PTT can open the window | Hybrid |

```bash
uv run alex --mode listen        # default for meetings
uv run alex --mode ptt           # desk demo with hotkey
uv run alex --text-input         # type instead of speak, audio out
uv run alex --text-input --silent  # text in / text out, no audio (Zoom-safe)
```

## What's in the box

**Speech pipeline (local)**
- `openWakeWord` (alexa placeholder) or PTT hotkey via `pynput` or continuous listening
- Silero VAD + Pipecat smart-turn analyzer for end-of-utterance detection
- MLX-Whisper `large-v3-turbo-Q4` with finance vocab biasing
- `AlwaysUserMuteStrategy` for self-echo guard (until hardware AEC is wired)

**Reasoning + tools (cloud)**
- **Tier 1**: Bedrock Claude Haiku 4.5 with prompt caching
- **Tier 2**: Bedrock Claude Sonnet 4.5 via `escalate_to_sonnet` tool
- **Filler**: pre-canned "Let me check…" phrases via Polly on every tool call

**Memory**
- 5-minute rolling buffer of all room transcripts
- Injected as context on every triggered LLM call
- VAD-driven follow-up window so natural conversation flows

**Retrieval**
- `rag_lookup`: LanceDB hybrid (BM25 + dense via bge-m3) over PDFs / markdown
- `sql_lookup`: DuckDB read-only SQL over structured allocation / time-series / factor / indicator data
- `entity_lookup`: hand-curated JSON of ticker → asset-class / fund → holdings / factor → tickers
- `web_search`: OpenAI Responses API with `web_search_preview` (when `OPENAI_API_KEY` is set)

**Output (cloud)**
- Amazon Polly Generative streaming TTS (Ruth by default)

**Observability**
- Per-turn JSONL logs in `latency_runs/` with `stt / llm_start / llm_ttft / llm_done / tts_ttfa / perceived`
- `uv run python -m alex.instrumentation latency_runs/<file>` for p50/p95 summary
- Emoji-prefixed log markers: 👂 passive, ⚡ triggered, 💬 follow-up, 🔎 RAG, 🛢 SQL, 🎓 Sonnet, 💬 filler, 🗣 said, ⏱ turn

## Pre-meeting setup

```bash
# Drop research PDFs / .md / .txt in corpora/ (anywhere; recursive).
# Drop real data in corpora/private/ — it's gitignored.
uv run python -m ingest_cli.pdfs build --fresh

# Optionally edit ingest_cli/datapoints.py with real allocations / time-series,
# then re-seed:
uv run python -m ingest_cli.datapoints seed --fresh

# Edit corpora/vocab.yaml with the meeting's ticker list, fund names, and
# people's names so Whisper biases toward your domain vocabulary.

# Edit corpora/entities.json with your ticker→asset-class, fund→holdings,
# person→role mappings.
```

Ingest of ~1000 pages takes ~1 minute. Query latency stays under 100 ms p95 well past 100K chunks per LanceDB's NEON-optimized benchmarks.

## Architecture

See **[docs/architecture.md](docs/architecture.md)** for the full design rationale, the pipeline diagram, the LLM tier routing logic, and the latency budget.

Other docs:
- **[docs/audio-setup.md](docs/audio-setup.md)** — BlackHole + Audio MIDI Setup for the in-room deployment
- **[docs/aec-status.md](docs/aec-status.md)** — Why software AEC isn't wired (yet); hardware AEC workaround
- **[docs/configuration.md](docs/configuration.md)** — Every CLI flag and `.env` variable
- **[alex/wake/trainer/README.md](alex/wake/trainer/README.md)** — Docker pipeline to train a true "Hey Alex" wake-word model (not required for LISTEN mode)

## Known limitations

- **Sub-1000 ms perceived latency** for tool-using turns isn't hit today (1.7-3 s on warm Bedrock Haiku + Polly Generative). The speculative-classifier path that would close this gap is documented but unbuilt.
- **Software AEC** is blocked on Apple Silicon (every WebRTC/Speex package fails to build with `-mfpu=` flags). Use a USB conference speakerphone with hardware AEC (Jabra Speak2 75) and flip `ALEX_HARDWARE_AEC_PRESENT=true` for barge-in to work.
- **"Hey Alex"** as a literal wake word in `wakeword` mode isn't wired — the bundled `alexa_v0.1.onnx` fires on "Alexa". `listen` mode sidesteps this entirely with substring matching. Train a custom model via `alex/wake/trainer/` if you specifically want `--mode wakeword` with "Hey Alex".
- **30-minute conversation memory** is architecturally supported (Haiku has 200 K context) but hasn't been stress-tested for hour-long meetings. A sliding-window summarizer is the natural next step.

## Project layout

```
realtime-local-talk/
├── alex/                       # the runtime
│   ├── app.py                  # CLI entry point (typer)
│   ├── pipeline.py             # Pipecat pipeline assembly
│   ├── config.py               # pydantic settings (env-driven)
│   ├── instrumentation.py      # per-turn latency JSONL + summary CLI
│   ├── audio/routing.py        # CoreAudio device discovery + setup_check
│   ├── llm/
│   │   ├── tools.py            # ToolsSchema builder + RAG handler
│   │   ├── escalation.py       # Sonnet 4.5 escalation tool
│   │   └── web_search.py       # OpenAI Responses web_search tool
│   ├── rag/
│   │   ├── embedder.py         # MLX bge-m3 4-bit
│   │   ├── lancedb_store.py    # hybrid BM25 + dense retrieval
│   │   ├── duckdb_store.py     # structured / time-series queries
│   │   ├── entities.py         # entity-relationship JSON loader
│   │   └── vocab.py            # Whisper initial_prompt builder
│   ├── stt/mlx_whisper_service.py  # vocab-biased Whisper subclass
│   ├── tts/                    # (Polly via Pipecat; module reserved)
│   ├── turn/
│   │   ├── keyword_router.py   # LISTEN trigger + follow-up window
│   │   └── transcript_buffer.py # rolling N-min meeting memory
│   └── wake/
│       ├── openwakeword_runner.py  # ONNX wake-word detector
│       ├── ptt.py              # global PTT hotkey via pynput
│       └── trainer/            # Docker pipeline for custom "Hey Alex"
├── ingest_cli/
│   ├── pdfs.py                 # PDF / markdown ingest → LanceDB
│   └── datapoints.py           # synthetic structured data → DuckDB
├── corpora/                    # config + sample notes (tracked) + private/ (gitignored)
├── tests/                      # unit tests for harnesses + router
├── docs/                       # architecture, audio setup, AEC status, configuration
└── pyproject.toml              # uv + hatchling
```

## Stack credits

- [Pipecat](https://github.com/pipecat-ai/pipecat) — voice pipeline framework
- [MLX](https://github.com/ml-explore/mlx) + [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) + [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings) — Apple Silicon ML runtime
- [LanceDB](https://github.com/lancedb/lancedb) — embedded NEON-tuned vector + FTS store
- [DuckDB](https://duckdb.org/) — embedded analytics DB
- [openWakeWord](https://github.com/dscripka/openWakeWord) — bundled wake-word models (placeholder until custom training)
- [pynput](https://github.com/moses-palmer/pynput) — global keyboard hotkey
- AWS Bedrock (Anthropic Claude 4.5, Amazon Polly Generative)
- OpenAI Responses API (web_search_preview)
