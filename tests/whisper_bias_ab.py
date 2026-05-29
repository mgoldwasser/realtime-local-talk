"""A/B test for Whisper's ``condition_on_previous_text``.

Round-trip: Polly TTS → wav → mlx_whisper.transcribe(both settings) → compare.

This is not a pytest — it's a CLI you run when you want to make a decision.

    uv run python tests/whisper_bias_ab.py

Decision criteria:
- Word error rate (WER) on the known transcript
- Repetition rate (Whisper's known failure mode with ``condition_on_previous_text=True``)
"""

from __future__ import annotations

import re
import tempfile
from collections import Counter
from pathlib import Path

import boto3
import mlx_whisper
import numpy as np
import soundfile as sf
from dotenv import load_dotenv

load_dotenv()

MODEL = "mlx-community/whisper-large-v3-turbo-q4"
POLLY_VOICE = "Ruth"
POLLY_ENGINE = "generative"

# Sentences chosen for finance domain + a couple of multi-sentence/pause
# patterns that historically trigger the repetition hallucination.
TEST_SENTENCES = [
    # Single short sentences (baseline).
    "We are modestly overweight emerging markets equities for the first quarter.",
    "The Fed held the target range at four and a quarter to four and a half percent.",
    "High yield spreads are at three hundred eight basis points, eighteenth percentile since 2010.",
    # Sentences with pauses + parallel clauses (repetition risk).
    "The relationship between interest rates, inflation, and asset valuations is fundamental to portfolio construction.",
    "Higher rates compress equity multiples, raise borrowing costs, and reduce the present value of future cash flows.",
    # Sentences with tickers + numeric content (vocab + repetition).
    "SPY, IWM, and EFA — we trimmed exposure across all three this quarter.",
    "Duration is 6.1 years against a policy benchmark of 5.7 years.",
]

# Vocab bias prompt — matches what the live system uses.
VOCAB_PROMPT = (
    "Vocabulary — tickers: SPY, IWM, EFA, AGG, TLT, HYG, LQD; "
    "terms: basis points, Treasury yields, investment grade, "
    "duration, FOMC, federal reserve."
)


def polly_synth(text: str) -> np.ndarray:
    """Return 16 kHz float32 mono samples for ``text`` via Polly Generative."""
    polly = boto3.client("polly", region_name="us-east-1")
    resp = polly.synthesize_speech(
        Text=text,
        VoiceId=POLLY_VOICE,
        Engine=POLLY_ENGINE,
        OutputFormat="pcm",
        SampleRate="16000",
    )
    raw = resp["AudioStream"].read()
    # PCM is little-endian signed 16-bit; convert to float32 [-1, 1].
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe(audio: np.ndarray, *, condition_on_previous_text: bool) -> str:
    out = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=MODEL,
        temperature=0.0,
        language="en",
        initial_prompt=VOCAB_PROMPT,
        condition_on_previous_text=condition_on_previous_text,
    )
    return " ".join(seg["text"].strip() for seg in out.get("segments", []))


# Light WER — token-level edit distance using dynamic programming.
def _wer(ref: str, hyp: str) -> tuple[float, int, int, int]:
    """Return (wer, substitutions, deletions, insertions)."""

    def _normalize(s: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", s.lower())

    r = _normalize(ref)
    h = _normalize(hyp)
    n, m = len(r), len(h)
    if n == 0:
        return (float("inf") if m else 0.0, 0, 0, m)
    # DP for edit distance with backtrace counts.
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if r[i - 1] == h[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])
    # Backtrace for individual S/D/I counts.
    i, j = n, m
    s = d = ins = 0
    while i > 0 and j > 0:
        if r[i - 1] == h[j - 1]:
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j - 1] + 1:
            s += 1
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j] + 1:
            d += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    d += i
    ins += j
    return ((s + d + ins) / n, s, d, ins)


def _repetition_score(text: str) -> int:
    """How many word-bigrams repeat at least twice? Higher = more hallucinated repetition."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    if len(tokens) < 2:
        return 0
    bigrams = list(zip(tokens, tokens[1:]))
    counts = Counter(bigrams)
    return sum(c for c in counts.values() if c >= 2)


def main() -> None:
    print(f"Round-trip: Polly Generative ({POLLY_VOICE}) → mlx-whisper {MODEL.split('/')[-1]}\n")

    results: list[dict] = []
    for i, ref in enumerate(TEST_SENTENCES, start=1):
        print(f"[{i}/{len(TEST_SENTENCES)}] {ref!r}")
        audio = polly_synth(ref)
        hyp_on = transcribe(audio, condition_on_previous_text=True)
        hyp_off = transcribe(audio, condition_on_previous_text=False)
        wer_on, *_ = _wer(ref, hyp_on)
        wer_off, *_ = _wer(ref, hyp_off)
        rep_on = _repetition_score(hyp_on)
        rep_off = _repetition_score(hyp_off)
        print(f"   ON : wer={wer_on:.2%} rep={rep_on}  → {hyp_on!r}")
        print(f"   OFF: wer={wer_off:.2%} rep={rep_off}  → {hyp_off!r}")
        results.append(
            dict(wer_on=wer_on, wer_off=wer_off, rep_on=rep_on, rep_off=rep_off)
        )
        print()

    # Summary.
    n = len(results)
    mean_wer_on = sum(r["wer_on"] for r in results) / n
    mean_wer_off = sum(r["wer_off"] for r in results) / n
    sum_rep_on = sum(r["rep_on"] for r in results)
    sum_rep_off = sum(r["rep_off"] for r in results)
    print("=" * 60)
    print(f"  mean WER   ON : {mean_wer_on:.2%}")
    print(f"  mean WER   OFF: {mean_wer_off:.2%}")
    print(f"  total reps ON : {sum_rep_on}")
    print(f"  total reps OFF: {sum_rep_off}")
    print()
    if mean_wer_off < mean_wer_on - 0.01 or sum_rep_off < sum_rep_on:
        print("⇒ recommend condition_on_previous_text=False")
    elif mean_wer_on < mean_wer_off - 0.01:
        print("⇒ keep condition_on_previous_text=True (current default)")
    else:
        print("⇒ within noise — keep current default")


if __name__ == "__main__":
    main()
