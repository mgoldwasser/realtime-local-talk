# Training a custom "Hey Alex" wake word

openWakeWord's training pipeline runs only on Linux x86_64 (Piper TTS +
tensorflow-lite quirks), so we do it in Docker.

The repo ships with `alex/wake/openwakeword_runner.py` configured to use
openWakeWord's bundled `alexa_v0.1.onnx` as a placeholder — phonetically
close to "Alex" and good enough for early testing. Train a real model
when you want fewer false positives from "Alexander", "Alexandra", etc.

## Quick start

```bash
# From repo root.
docker build -t alex-wakeword-trainer alex/wake/trainer/
mkdir -p models
docker run --rm -v "$(pwd)/models:/out" alex-wakeword-trainer
```

Training takes 30-60 minutes depending on host CPU (we don't currently
use a GPU — Docker on macOS doesn't get Metal). Output lands at
`models/hey_alex_v1.onnx`.

## Wire the trained model into the runtime

Set the env var in `.env` (or pass on the command line):

```bash
ALEX_WAKEWORD_MODEL_PATH=/Users/<you>/realtime-local-talk/models/hey_alex_v1.onnx
```

…then:

```bash
uv run alex --mode wakeword
```

## Tuning the recipe

`recipe.yml` is the only file you should edit for retraining:

- **n_samples** — More positives → better recall, longer training.
- **tts_voices** — Add more Piper voices for accent diversity. Browse
  https://huggingface.co/rhasspy/piper-voices for options.
- **hard_negatives** — Audio your meeting context will contain that
  should *not* trigger. Add committee member first names that sound
  similar to "Alex" here.
- **augmentation_rounds** — Noise / reverb / pitch variations.
  3 is the sweet spot; 1 will overfit, 5 wastes time.

After editing, rebuild and rerun:

```bash
docker build -t alex-wakeword-trainer alex/wake/trainer/
docker run --rm -v "$(pwd)/models:/out" alex-wakeword-trainer
```

## What's actually inside the trained model

openWakeWord splits the architecture two ways:

1. **A shared melspectrogram + speech-embedding model** (bundled, frozen).
   This converts raw audio → 128-dim embeddings every 80 ms.
2. **A small per-wakeword classifier** (the file we train). Takes a
   sequence of embeddings and produces a confidence score.

You're only training the second piece. That's why training is fast and
the output `.onnx` is small (~50 KB).
