# Alex — Real-Time Local Voice Meeting Assistant

Hands-free voice assistant for in-room meetings. Runs on Apple Silicon, hits sub-500 ms perceived latency by combining a local speech pipeline with AWS Bedrock LLMs and a "stall while escalating" routing pattern.

The full design is in `~/.claude/plans/i-want-to-create-whimsical-papert.md`.

## Setup

```bash
# 1. Install dependencies into a uv-managed venv
uv sync

# 2. Configure credentials
cp .env.example .env
# Edit .env: set AWS_PROFILE (or keys) and OPENAI_API_KEY.

# 3. Bedrock auto-enables serverless foundation models on first invoke (the
#    old "Model access" page is retired). First-time Anthropic users may be
#    prompted once to submit use-case details — do that in the Bedrock
#    console playground before the first programmatic call if you hit a
#    "submit use case" error from boto3.
```

## Run

Phase 1 (PTT, no wake word, no STT yet — text-in/voice-out smoke test):

```bash
uv run alex --text-input --tts polly --llm haiku
```

Phase 2+ (voice in/out):

```bash
uv run alex --mode ptt --tts polly --llm haiku
uv run alex --mode wakeword                # after Phase 3
```

## Latency

Every turn emits structured timing to `latency_runs/<timestamp>.jsonl`. Inspect with:

```bash
uv run python -m alex.instrumentation summary latency_runs/<file>
```

## Project layout

See the plan file for the full layout. Top-level:

- `alex/` — the runtime app
- `ingest_cli/` — offline corpus ingestion
- `corpora/` — gitignored; drop research PDFs here
- `tests/` — unit + latency tests
