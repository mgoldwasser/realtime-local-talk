"""Compose a Whisper ``initial_prompt`` from a YAML vocab file.

Whisper's prompt budget is ~244 tokens — roughly 150-180 words. The loader
truncates if the file gets too verbose; emit a warning when that happens
so the user knows their tail vocab isn't biasing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml
from loguru import logger

# Conservative cap; leaves headroom for Whisper's internal context.
_MAX_PROMPT_CHARS = 900


def load_vocab_prompt(path: Path | str | None) -> str | None:
    """Read a vocab YAML and return a single bias prompt, or ``None`` if missing.

    The YAML is expected to be a mapping of category name → list of strings.
    Categories are joined in file order; within each category, terms are
    comma-separated. Whisper biases on raw text so the framing doesn't matter
    much; we add a leading 'Vocabulary:' to nudge the model.
    """
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        logger.warning(f"vocab file not found: {path}")
        return None

    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        logger.warning(f"vocab file is not a mapping; ignoring: {path}")
        return None

    parts: list[str] = []
    for category, terms in raw.items():
        if not isinstance(terms, Iterable):
            continue
        clean = [str(t).strip() for t in terms if str(t).strip()]
        if clean:
            parts.append(f"{category}: " + ", ".join(clean))

    if not parts:
        return None

    prompt = "Vocabulary — " + "; ".join(parts) + "."
    if len(prompt) > _MAX_PROMPT_CHARS:
        logger.warning(
            f"vocab prompt truncated from {len(prompt)} to {_MAX_PROMPT_CHARS} chars"
        )
        prompt = prompt[:_MAX_PROMPT_CHARS].rsplit(",", 1)[0] + "."

    logger.info(f"vocab prompt: {len(prompt)} chars, {len(parts)} categories")
    return prompt
