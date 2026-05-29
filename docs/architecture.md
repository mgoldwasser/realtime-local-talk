# Architecture

How Alex actually works, and why each piece is the way it is.

## TL;DR

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              Mic (room audio)                            │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │ 16 kHz int16 mono via PortAudio
                                     ▼
                       ┌──────────────────────────┐
                       │  LocalAudioTransport      │  Pipecat input
                       │  + Silero VAD analyzer    │
                       └──────────────┬───────────┘
                                      │
                       ┌──────────────▼───────────┐
                       │  AlwaysUserMuteStrategy   │  drops audio frames
                       │  (self-echo guard)        │  while Alex is speaking
                       └──────────────┬───────────┘
                                      │
                       ┌──────────────▼───────────┐
                       │  BiasedMLXWhisperSTT      │  vocab-biased Whisper Q4
                       │  on Apple Neural Engine   │  → TranscriptionFrame
                       └──────────────┬───────────┘
                                      │
                       ┌──────────────▼───────────┐
                       │  TranscriptKeywordRouter  │  LISTEN-mode gate
                       │  + RollingTranscriptBuffer│  + VAD follow-up window
                       └──────────────┬───────────┘
                                      │  only triggered or follow-up frames
                                      │  pass downstream; rewritten with
                                      │  recent room context prepended
                       ┌──────────────▼───────────┐
                       │  LLMContextAggregatorPair │  builds user message
                       └──────────────┬───────────┘
                                      │
                       ┌──────────────▼───────────┐  Tools registered:
                       │  AWSBedrockLLMService     │   • rag_lookup
                       │  Claude Haiku 4.5 +       │   • sql_lookup
                       │  prompt caching           │   • entity_lookup
                       │                           │   • web_search
                       │                           │   • escalate_to_sonnet
                       └──────────────┬───────────┘
                                      │ on_function_calls_started →
                                      │ speculative filler ("Let me check…")
                                      ▼
                       ┌──────────────────────────┐
                       │  AWSPollyTTSService       │  Polly Generative streaming
                       │  Ruth (en-US)             │  → 24 kHz PCM audio
                       └──────────────┬───────────┘
                                      ▼
                       ┌──────────────────────────┐
                       │   LocalAudioTransport     │  Pipecat output → speakers
                       └──────────────────────────┘
```

Per-stage latency taps run alongside this and emit JSONL to `latency_runs/`.

## Why this shape

### Always-transcribing > wake-word for meeting use

The first version of Alex used a Picovoice/openWakeWord ONNX detector. That made sense for phones and Echo devices where battery and microphone privacy are the dominant constraints. It made *no* sense for a plugged-in MacBook running a meeting assistant:

1. The use case requires a rolling transcript anyway (so Alex can answer questions like "what did Charles just say about Treasuries?")
2. Once you're running Whisper continuously, wake-word detection collapses to a one-line substring check on the transcript
3. The wake-word approach is brittle for new trigger phrases — "Hey Alex" needs training, "Alex" alone fires on neighbors like "Alexander"

LISTEN mode replaces ONNX wake-word detection with:
- MLX Whisper running on every VAD-bounded utterance (continuous)
- `TranscriptKeywordRouter` doing case-insensitive substring match on the finalized transcript
- A 5-minute `RollingTranscriptBuffer` of everything said (triggered or not)
- A VAD-driven follow-up window so natural back-and-forth doesn't need re-addressing

Compute cost is well under 15% of one ANE core on M3 during active speech; idle is ~0%. Transcript memory for a 30-minute meeting is < 5 MB.

### Self-echo before AEC

Without acoustic echo cancellation, Polly playing through the laptop speakers gets re-transcribed by Whisper, and "what Alex just said" looks like a new user turn. Classic feedback loop.

Pipecat ships `AlwaysUserMuteStrategy` which gates user input frames whenever `BotStartedSpeakingFrame` → `BotStoppedSpeakingFrame` is in flight. It's the default in voice modes. Tradeoff: no barge-in until either hardware AEC (USB conference speakerphone with built-in DSP, e.g. Jabra Speak2 75) or software AEC3 lands. See `docs/aec-status.md` for why software AEC3 is currently blocked on Apple Silicon arm64.

### Tool-routed LLM > monolithic prompt

Stuffing every relevant document into one Haiku prompt would work but waste tokens and confuse the model on mixed queries. Instead:

- `rag_lookup` (LanceDB hybrid BM25 + dense) → unstructured research notes
- `sql_lookup` (DuckDB read-only) → current allocations, returns, factor z-scores, indicators
- `entity_lookup` (hand-curated JSON) → ticker / fund / sector / factor relationships
- `web_search` (OpenAI Responses) → current external information
- `escalate_to_sonnet` (Bedrock Claude Sonnet 4.5) → multi-step reasoning the cheap model would botch

Haiku decides per turn which tools to call (often multiple in sequence). The system prompt nudges it: SQL for numbers, RAG for narrative, entity for relationships, web for recent.

This is one place where Anthropic's tool-use API really earns its keep — Haiku is good at picking the right tool and chaining when needed.

### Speculative filler instead of waiting

The first tool-using turn takes ~2-4 s end to end:

| Stage | ms (typical) |
|---|---|
| STT finalize | 50-150 |
| Haiku TTFT to tool-use decision | 700-1200 |
| Tool execution (RAG / SQL) | 5-100 |
| Haiku TTFT on second call with tool result | 700-1200 |
| Polly TTFA | 150-300 |
| Total perceived | 1700-3000 |

`on_function_calls_started` is a Pipecat event that fires the moment Haiku emits its tool-use decision. We hook it to queue `TTSSpeakFrame("Let me check…")` (chosen from 6 phrasings). Polly streams the filler in ~150 ms, and the real answer arrives ~2 s later — by which point Polly is just finishing the filler and transitions cleanly into the actual reply.

Net perceived latency feels like ~250 ms on a tool-using turn, vs the raw 3 s.

(The plan called for an even tighter design — a fast classifier that predicts "this needs a tool" *before* calling Haiku, firing the filler at ~200 ms regardless of Haiku's decision time. That's documented in `~/.claude/plans/i-want-to-create-whimsical-papert.md` as future work.)

### Why Sonnet via tool call instead of LLM swap

The plan's original design had a Tier-0 micro-classifier deciding Haiku vs Sonnet per turn. That requires either (a) running a small local LLM as a router, or (b) using Pipecat's `LLMSwitcher` to dynamically swap LLM services mid-pipeline.

The shipped version is simpler: register `escalate_to_sonnet` as a tool. Haiku, given a system prompt that explicitly says when to escalate (multi-step reasoning, scenario analysis, comparing trade-offs, "why" / "explain"), decides per turn. When it calls the tool:

1. Filler fires immediately (via the same `on_function_calls_started` hook)
2. The tool runs Sonnet 4.5 via `bedrock-runtime.converse()` directly
3. Sonnet's answer comes back as the tool result
4. Haiku relays it in conversational voice

Cost: Haiku's relay step adds ~2-4 s and a small token cost. Benefit: zero pipeline restructuring, easy to extend with more tier-2 models (GPT-5, Opus). Future polish: stream Sonnet straight to TTS and skip Haiku's relay.

### Latency budget (LISTEN mode, warm Bedrock)

| Stage | Median |
|---|---|
| VAD endpoint + Pipecat smart-turn | ~150 ms |
| MLX Whisper finalize (M3 ANE) | 80-150 ms |
| Bedrock Haiku TTFT (cold) | 1000-1500 ms |
| Bedrock Haiku TTFT (warm) | 700-900 ms |
| Tool dispatch (rag / sql / entity) | 5-80 ms |
| Bedrock Haiku second-call TTFT | similar |
| Polly Generative TTFA | 150-300 ms |
| **Total perceived (chitchat)** | **~1.7 s** |
| **Total perceived (tool-using, with filler)** | **~250 ms to filler, ~3 s to real answer** |
| **Total perceived (Sonnet escalation, with filler)** | **~250 ms to filler, ~12 s to real answer** |

Sub-500 ms perceived on tool-using turns is achievable with the speculative classifier path; not implemented today.

## Activation modes side-by-side

| Mode | Mic open when | Filter | Audio gate | Best for |
|---|---|---|---|---|
| **`listen`** | always | substring on transcript | `AlwaysUserMuteStrategy` while bot speaks | Meeting use |
| `wakeword` | always (detector reads everything) | `WakeWordMuteStrategy` outside listen window + `AlwaysUserMuteStrategy` | both | Battery-sensitive |
| `ptt` | only while toggled on | `PTTMuteStrategy` + `AlwaysUserMuteStrategy` | both | Desk demo |
| `both` | wake word OR PTT opens window | `WakeWordMuteStrategy` AND `PTTMuteStrategy` (open if either says open) | + `AlwaysUserMuteStrategy` | Hybrid |
| `text-input` | n/a | keyword router still applies in LISTEN | none | Headless testing |

`AlwaysUserMuteStrategy` is the self-echo guard and is added by default to every voice mode unless `ALEX_HARDWARE_AEC_PRESENT=true`.

## The follow-up window (LISTEN mode)

After Alex finishes speaking, a natural conversation looks like:

> User: *Hey Alex, what's our EM view?*
> Alex: *We're modestly overweight, plus 100 bps. Want me to break down the thesis?*
> User: *Yes please.*  ← no "Hey Alex"!
> Alex: *Three drivers: dollar peaking, China policy easing, EM earnings revisions widening…*

The bare-trigger approach would miss "Yes please" because it lacks the keyword. The follow-up window solves it:

- `BotStoppedSpeakingFrame` opens the window with `follow_up_initial_secs` (default 10) of grace
- Each `UserStartedSpeakingFrame` while inside the window extends the deadline by `follow_up_extend_secs` (default 6)
- `BotStartedSpeakingFrame` closes the window cleanly when Alex starts a new reply

So the window naturally extends as long as someone is actively responding, and only closes on sustained silence. Default behavior:

- Alex stops → 10 s window opens
- User starts replying at 8 s in → deadline pushed to "now + 6 s"
- User pauses 3 s mid-sentence to think → still inside window
- User resumes → another extend
- User finishes, no further speech → window closes after the 6 s tail-out

## RAG layer details

**Why LanceDB**: NEON-tuned SIMD on Apple Silicon (3.5× speedup over naive on M-series per their own benchmarks). Hybrid BM25 + dense in one process. Embedded (no server). Sub-100 ms p95 at 1B vectors in their published benchmarks; well under 10 ms at 100 K chunks for our use case.

**Why bge-m3 over Cohere**: speed. bge-m3 4-bit on MLX takes 6-12 ms per query. Cohere Embed v4 via Bedrock takes 60-90 ms. ~60 ms saved per turn is meaningful when total latency budget is ~1 s.

**Why hybrid retrieval**: BM25 catches exact-term matches (tickers, fund names, proper nouns) that dense retrieval can miss. RRF fusion (default) merges the two rankings — no parameter tuning needed.

**Why DuckDB alongside LanceDB**: SQL is a much better tool than vector search for "what's our current allocation to EM?". The router lets Haiku pick the right tool per question, with `rag_lookup` + `sql_lookup` as separate function calls.

**Why a hand-curated entities JSON instead of GraphRAG**: GraphRAG / LightRAG / HippoRAG add 50-150 ms per turn and the auto-extracted entity quality is mediocre for finance domains. A hand-maintained mapping of "SPY → US Equity → S&P 500 → Quality + Momentum" is < 1 ms to look up and 100% accurate. It's also a single small file the user can edit directly.

## Vocab biasing

Whisper has an `initial_prompt` kwarg that conditions the decoder. It's not training data — it's a short text snippet (~244 token budget) that nudges decoding toward terms in it. We compose the prompt from `corpora/vocab.yaml`:

```yaml
tickers:    [SPY, IVV, IWM, EFA, EEM, TLT, ...]
funds:      [T. Rowe Price Blue Chip Growth, ...]
terms:      [basis points, Bloomberg US Aggregate, ...]
people:     [Sebastien Page, Charles Shriver, ...]
```

This dramatically improves accuracy on ticker mentions and proper nouns that the off-the-shelf Whisper model has never seen at scale.

## Where each piece runs

| Component | Location | Cost |
|---|---|---|
| Wake word (mode: wakeword) | local CPU | ~50 KB ONNX, micro-CPU |
| VAD | local CPU | tiny |
| Smart turn | local CPU | ~65 ms per turn |
| STT (MLX-Whisper Q4) | local ANE | ~80-150 ms per utterance |
| Embeddings (bge-m3 4-bit) | local ANE | 6-12 ms per query, ~340 MB |
| LanceDB vector + FTS | local disk + NEON | < 10 ms at 100 K chunks |
| DuckDB | local disk | < 5 ms typical query |
| Entities JSON | RAM | < 1 ms |
| LLM Haiku 4.5 | Bedrock (us-east-1) | ~700-1500 ms TTFT |
| LLM Sonnet 4.5 (escalation) | Bedrock (us-east-1) | ~1500-2500 ms TTFT |
| Web search | OpenAI Responses (when key set) | ~2-4 s typical |
| TTS Polly Generative | AWS (us-east-1) | ~150-300 ms TTFA, streaming |

Only the LLM query and the web search ever leave the laptop. The research corpus stays local. With `OPENAI_API_KEY` unset, web search is omitted entirely.

## Known things-not-built

See README "Known limitations" for the user-facing list. Internally:

- Speculative classifier that predicts tool-need before calling Haiku → unbuilt; would close most of the sub-1 s gap on tool turns
- Sonnet streaming straight to TTS (skipping Haiku relay) → unbuilt; would save 2-4 s on escalation turns
- Software AEC (WebRTC AEC3) → blocked on Apple Silicon arm64 build flags; documented in `docs/aec-status.md`
- Custom "Hey Alex" wake-word for `wakeword` mode → scaffolded in `alex/wake/trainer/` but not run; LISTEN mode sidesteps this
- Sliding-window summarization for meetings > ~1 hour (where 200 K Haiku context starts to matter) → unbuilt
- Persistent conversation memory across meetings → unbuilt
- Transcript archive integration for compliance (Smarsh / Theta Lake / Global Relay) → unbuilt
