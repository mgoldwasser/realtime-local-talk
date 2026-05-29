# Configuration reference

Every CLI flag and `.env` variable, plus what they do.

## CLI flags (`uv run alex …`)

| Flag | Values | Default | Effect |
|---|---|---|---|
| `--mode` | `listen` / `wakeword` / `ptt` / `both` | `listen` | Activation mode |
| `--text-input` | flag | off | Read stdin instead of mic |
| `--silent` | flag | off | Skip TTS / audio out (Zoom-safe debug) |
| `--tts` | `polly` / `kokoro` | `polly` | TTS backend (kokoro = local MLX fallback, not implemented yet) |
| `--llm` | `haiku` / `sonnet` / `gpt5` | `haiku` | Force a specific LLM tier (overrides per-turn routing) |
| `--help` | flag | — | List all flags |

## `.env` variables

All variables are read by `pydantic-settings` with prefix `ALEX_` (except AWS / OpenAI standard names). Defaults match `.env.example`.

### AWS

| Variable | Default | Notes |
|---|---|---|
| `AWS_REGION` | `us-east-1` | Bedrock + Polly region. Some models (e.g. latency-opt Haiku 3.5) only in us-east-2. |
| `AWS_PROFILE` | — | If set, boto3 uses this profile. Otherwise standard credential chain. |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN` | — | Alternative to profile-based auth. |

### OpenAI

| Variable | Default | Notes |
|---|---|---|
| `OPENAI_API_KEY` | — | Enables the `web_search` tool. Tool is omitted at startup if unset. |

### Activation

| Variable | Default | Notes |
|---|---|---|
| `ALEX_ACTIVATION` | `listen` | Same values as `--mode` flag |
| `ALEX_HARDWARE_AEC_PRESENT` | `false` | Set `true` only with a hardware-AEC speakerphone (Jabra Speak2 75 / Poly Sync 60). Disables software self-mute so barge-in works. |

### TTS

| Variable | Default | Notes |
|---|---|---|
| `ALEX_TTS` | `polly` | TTS backend |
| `ALEX_POLLY_VOICE` | `Ruth` | Any Polly Generative voice available in your region |

### LLM

| Variable | Default | Notes |
|---|---|---|
| `ALEX_LLM_HAIKU` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | Cross-region inference profile ID. Match Bedrock console. |
| `ALEX_LLM_SONNET` | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` | Same — used by `escalate_to_sonnet` tool |
| `ALEX_LLM_FILLER` | `us.amazon.nova-micro-v1:0` | Reserved for the Nova-based filler tier (unused today; speculative filler uses Polly directly) |
| `ALEX_LLM_LATENCY` | `standard` | `optimized` enables Bedrock latency-optimized inference. Not yet supported for Claude 4.5 in any region as of build time. |

### STT

| Variable | Default | Notes |
|---|---|---|
| `ALEX_STT_MODEL` | `LARGE_V3_TURBO_Q4` | One of `TINY` / `MEDIUM` / `LARGE_V3` / `LARGE_V3_TURBO` / `DISTIL_LARGE_V3` / `LARGE_V3_TURBO_Q4`. Q4 is the recommended balance of speed and quality on M3/M4. |

### Audio devices

| Variable | Default | Notes |
|---|---|---|
| `ALEX_INPUT_DEVICE` | (system default mic) | Substring match against PortAudio device names. Run `uv run python -m alex.audio.routing` to list. |
| `ALEX_OUTPUT_DEVICE` | (system default) | Same |

### Wake word (`--mode wakeword` / `both`)

| Variable | Default | Notes |
|---|---|---|
| `ALEX_WAKEWORD_MODEL_PATH` | — (bundled `alexa_v0.1.onnx`) | Custom-trained ONNX path. See `alex/wake/trainer/`. |
| `ALEX_WAKEWORD_THRESHOLD` | `0.5` | 0.0-1.0; lower = more sensitive (more false positives). Use the `wake score=` logs to tune. |
| `ALEX_WAKEWORD_LISTEN_SECS` | `10.0` | How long the mic stays open after a wake-word fire |

### LISTEN mode (`--mode listen`)

| Variable | Default | Notes |
|---|---|---|
| `ALEX_LISTEN_TRIGGERS` | `alex,hey alex,ask alex` | Comma-separated, case-insensitive substring matches. Empty entries ignored. |
| `ALEX_LISTEN_BUFFER_MINUTES` | `5.0` | How much room transcript is kept and injected as context on triggered turns |
| `ALEX_LISTEN_FOLLOW_UP_INITIAL_SECS` | `10.0` | Grace period right after Alex finishes speaking. Set to `0` to require the trigger phrase on every turn. |
| `ALEX_LISTEN_FOLLOW_UP_EXTEND_SECS` | `6.0` | Window extension per `UserStartedSpeakingFrame` while inside the window |

### Paths

These are normally fine at their defaults but can be overridden if you want to keep data outside the project tree.

| Variable | Default | Notes |
|---|---|---|
| `ALEX_PROJECT_ROOT` | (repo root) | Used to anchor the others |
| `ALEX_LATENCY_DIR` | `latency_runs/` | Per-turn JSONL telemetry |
| `ALEX_VOCAB_PATH` | `corpora/vocab.yaml` | Whisper biasing prompt source |
| `ALEX_LANCE_DB_PATH` | `data/lance/` | Vector + FTS store |
| `ALEX_LANCE_TABLE` | `chunks` | LanceDB table name |
| `ALEX_DUCK_DB_PATH` | `data/duck.db` | DuckDB file |
| `ALEX_ENTITIES_PATH` | `corpora/entities.json` | Entity-relationship JSON |

## Ingest CLIs

```bash
# PDFs / .md / .txt in corpora/* → LanceDB
uv run python -m ingest_cli.pdfs build [--corpus-dir corpora] [--db data/lance] \
                                       [--table chunks] [--fresh]
uv run python -m ingest_cli.pdfs query "your search query" [-k 6]

# Synthetic structured data → DuckDB
uv run python -m ingest_cli.datapoints seed [--db data/duck.db] [--fresh] [--seed 42]
uv run python -m ingest_cli.datapoints query "SELECT * FROM current_allocations" [--db ...]
```

`--fresh` wipes the target store before ingesting. Use it to re-seed when you change `corpora/` content or the synthetic data generator.

## Tuning recipes

**Make Alex more responsive in follow-up:**
```
ALEX_LISTEN_FOLLOW_UP_INITIAL_SECS=15
ALEX_LISTEN_FOLLOW_UP_EXTEND_SECS=10
```

**Strict trigger required on every turn (no follow-up):**
```
ALEX_LISTEN_FOLLOW_UP_INITIAL_SECS=0
ALEX_LISTEN_FOLLOW_UP_EXTEND_SECS=0
```

**Reduce false triggers from "Alexander" / "Alexandra":**
```
ALEX_LISTEN_TRIGGERS=hey alex,ask alex
```

**Use a different voice:**
```
ALEX_POLLY_VOICE=Matthew
```

**Increase recall vs precision on wake word:**
```
ALEX_WAKEWORD_THRESHOLD=0.35
```

**Hardware AEC mode (Jabra Speak2 75 etc.) — enable barge-in:**
```
ALEX_HARDWARE_AEC_PRESENT=true
```

**Run on a model other than Q4 turbo for STT (larger = more accurate, slower):**
```
ALEX_STT_MODEL=LARGE_V3
```
