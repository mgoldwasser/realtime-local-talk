"""Tier-2 (Sonnet 4.5) escalation via a tool call from Tier-1 (Haiku 4.5).

Pattern: Haiku decides a question needs deeper reasoning, calls the
``escalate_to_sonnet`` tool, the existing ``on_function_calls_started``
hook plays a "give me a second" filler, the tool spins Sonnet 4.5 on
Bedrock directly, and the answer comes back as the tool result for
Haiku to relay. Adds ~1-2s but lands on the Sonnet-grade reasoning
that hard TAA questions need.

This is the simpler-than-LLM-switching approach. A future polish pass
can short-circuit Haiku's relay step and stream Sonnet straight to TTS.
"""

from __future__ import annotations

import asyncio
from typing import Any

import boto3
from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.services.llm_service import FunctionCallParams

ESCALATE_SCHEMA = FunctionSchema(
    name="escalate_to_sonnet",
    description=(
        "Escalate a complex question to Claude Sonnet 4.5 for deeper reasoning. "
        "Use this for: multi-step finance reasoning, scenario analysis, "
        "comparing several positions or trade-offs, questions where the user "
        "explicitly asks 'why' or 'explain' on a non-trivial topic, or anything "
        "you're not confident answering in one to three short sentences. "
        "Do NOT use for simple factual lookups, definitions, or chitchat — "
        "those you should answer directly. After calling, briefly relay the "
        "result back to the user in conversational voice."
    ),
    properties={
        "question": {
            "type": "string",
            "description": (
                "The question or reasoning task to delegate. Be specific — "
                "Sonnet only sees what you send, not the full conversation."
            ),
        },
        "context": {
            "type": "string",
            "description": (
                "Optional: any context from earlier turns or the RAG corpus "
                "that Sonnet should know about. Keep under 500 words."
            ),
        },
    },
    required=["question"],
)


SONNET_SYSTEM = (
    "You are Claude Sonnet 4.5 acting as a deep-reasoning oracle for a "
    "voice meeting assistant. Answer in a tight, conversational 2-4 "
    "sentence response that the assistant can read aloud. Lead with the "
    "answer; brief reasoning second. Do not use markdown or citations."
)


def make_escalate_handler(model_id: str, region: str):
    """Return a Pipecat function handler that runs Sonnet 4.5 via Bedrock."""

    # Lazily create the client — saves a few hundred ms at startup if never used.
    _client: list = []

    def _get_client():
        if not _client:
            _client.append(boto3.client("bedrock-runtime", region_name=region))
        return _client[0]

    async def escalate(params: FunctionCallParams) -> None:
        q = (params.arguments.get("question") or "").strip()
        ctx = (params.arguments.get("context") or "").strip()
        if not q:
            await params.result_callback({"error": "question is required"})
            return

        user_msg = q if not ctx else f"Context:\n{ctx}\n\nQuestion: {q}"

        def _run() -> str:
            client = _get_client()
            resp = client.converse(
                modelId=model_id,
                system=[{"text": SONNET_SYSTEM}],
                messages=[{"role": "user", "content": [{"text": user_msg}]}],
                inferenceConfig={"maxTokens": 400, "temperature": 0.3},
            )
            # Extract text out of the converse response.
            parts = resp["output"]["message"]["content"]
            return " ".join(p.get("text", "") for p in parts if "text" in p).strip()

        try:
            text = await asyncio.to_thread(_run)
        except Exception as e:
            logger.error(f"sonnet escalation failed: {e}")
            await params.result_callback({"error": str(e)})
            return

        logger.info(f"🎓 sonnet escalation completed ({len(text)} chars)")
        await params.result_callback({"answer": text})

    return escalate
