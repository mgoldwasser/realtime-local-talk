"""Alex CLI entry point.

Phase 1: text-input → Bedrock Haiku → Polly streaming → speakers.
Wake word, STT, RAG come online in later phases (Phases 2-4 in the plan).
"""

from __future__ import annotations

import asyncio
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console

from alex.config import Activation, LlmTier, Settings, TtsBackend, get_settings

console = Console()


def cli() -> None:
    """Typer entrypoint. Wraps ``main`` so ``alex --help`` and arg parsing work."""
    typer.run(main)


def main(
    text_input: bool = typer.Option(
        False,
        "--text-input",
        help="Read prompts from stdin instead of the mic. Useful in Phase 1 before STT is wired.",
    ),
    silent: bool = typer.Option(
        False,
        "--silent",
        help="Skip TTS/audio output. Useful for headless debugging on a call.",
    ),
    mode: Optional[Activation] = typer.Option(
        None, "--mode", help="Activation mode override (ptt | wakeword | both)."
    ),
    tts: Optional[TtsBackend] = typer.Option(
        None, "--tts", help="TTS backend override (polly | kokoro)."
    ),
    llm: Optional[LlmTier] = typer.Option(
        None, "--llm", help="Force a specific LLM tier (haiku | sonnet | gpt5)."
    ),
) -> None:
    load_dotenv()
    settings = get_settings()
    if mode is not None:
        settings.activation = mode
    if tts is not None:
        settings.tts = tts

    console.rule("[bold]Alex — Real-Time Voice Meeting Assistant[/bold]")
    console.print(f"region:     {settings.aws_region}")
    console.print(f"activation: {settings.activation.value}")
    console.print(f"tts:        {settings.tts.value} ({settings.polly_voice})")
    console.print(f"llm:        {(llm or LlmTier.HAIKU).value}")
    console.print(f"input:      {'text (stdin)' if text_input else 'microphone'}")
    console.print(f"output:     {'silent (text only)' if silent else 'audio'}")
    console.rule()

    from alex.pipeline import run_pipeline

    asyncio.run(
        run_pipeline(
            settings=settings, text_input=text_input, forced_tier=llm, silent=silent
        )
    )


if __name__ == "__main__":
    cli()
