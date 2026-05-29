"""Train a custom 'Hey Alex' openWakeWord model.

Runs inside the Linux Docker container only — the Piper + tensorflow-lite
stack is fragile on Apple Silicon.

Pipeline:
    1. Synthesize ~5000 positive "Hey Alex" samples via Piper TTS across
       multiple voices, speeds, and prosodies (per recipe.yml).
    2. Augment with noise / reverb / pitch (3 rounds).
    3. Generate hard-negative samples ("Alexa", "Alexander", etc.).
    4. Combine with openWakeWord's bundled negative corpus.
    5. Train the small CNN classifier on top of the shared embedding model.
    6. Export ONNX to /out/hey_alex_v1.onnx.

The intent is to be the canonical script you run once per major recipe
change. Re-run when the recipe yields too many false positives in real use.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

OUT_DIR = Path(os.environ.get("WAKEWORD_OUT", "/out"))
RECIPE = Path(__file__).parent / "recipe.yml"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(RECIPE) as f:
        recipe = yaml.safe_load(f)

    print(f"Training '{recipe['output_model_name']}' from recipe {RECIPE}")
    print(f"  target phrases : {recipe['target_phrase']}")
    print(f"  n_samples      : {recipe['n_samples']}")
    print(f"  voices         : {recipe['tts_voices']}")
    print(f"  hard negatives : {recipe['hard_negatives']}")
    print(f"  output dir     : {OUT_DIR}")

    # openWakeWord ships a high-level training API; we drive it directly here.
    # Import is at top of function so the module loads cleanly outside Docker
    # for unit testing the harness.
    from openwakeword.train import train_custom_model  # type: ignore[attr-defined]

    train_custom_model(
        target_phrase=recipe["target_phrase"],
        tts_voices=recipe["tts_voices"],
        n_positive_samples=recipe["n_samples"],
        augmentation_rounds=recipe["augmentation_rounds"],
        hard_negative_phrases=recipe["hard_negatives"],
        use_default_negatives=recipe["use_default_negatives"],
        output_dir=str(OUT_DIR),
        model_name=recipe["output_model_name"],
    )

    onnx_path = OUT_DIR / f"{recipe['output_model_name']}.onnx"
    print(f"\nDone. Model written to: {onnx_path}")
    print("\nNext step (host machine):")
    print(f"  cp models/{recipe['output_model_name']}.onnx <your local models dir>")
    print(f"  set ALEX_WAKEWORD_MODEL_PATH=models/{recipe['output_model_name']}.onnx")


if __name__ == "__main__":
    main()
