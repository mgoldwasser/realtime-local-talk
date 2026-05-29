"""``web_search`` tool — routes through OpenAI's Responses API with the
built-in web_search tool.

Pattern: Bedrock Claude (Haiku/Sonnet) decides we need fresh external info,
calls ``web_search`` with a focused query, the tool fires an OpenAI Responses
call that internally browses the web and returns a synthesized answer with
citations. Bedrock Claude then relays it in conversational voice.

Why OpenAI and not Anthropic's native web_search on Bedrock:

  As of 2026-05-29, AWS Bedrock advertises ``web_search_20250305`` in its
  Converse / InvokeModel validation schema (it appears in the list of
  accepted ``type`` values for ``additionalModelRequestFields.tools``)
  but actual invocation fails with a generic ``ValidationException:
  The provided request is not valid`` on both Haiku 4.5 and Sonnet 4.5.
  Tested with and without ``max_uses``, via Converse and InvokeModel,
  using inference profile IDs. This appears to be an AWS rollout gap
  — schema advertises support, runtime doesn't. Re-probe periodically;
  if it starts returning real results, the swap is trivial.

Tavily / Perplexity / Brave are out of our vendor allowlist.
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional

from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.services.llm_service import FunctionCallParams

WEB_SEARCH_SCHEMA = FunctionSchema(
    name="web_search",
    description=(
        "Search the live web for current information the model doesn't already "
        "know — recent market news, fresh economic data, today's prices, news "
        "from the past few days. Returns a synthesized answer with citations. "
        "Use when the user asks about anything time-sensitive ('today', 'this "
        "morning', 'latest', 'just announced') or current prices/levels. "
        "Do NOT use for evergreen finance concepts the model knows, or for "
        "questions answered by the research notes (use rag_lookup) or the "
        "structured database (use sql_lookup)."
    ),
    properties={
        "query": {
            "type": "string",
            "description": "Focused search query. Keep it short and specific.",
        },
    },
    required=["query"],
)


def make_web_search(
    model: str = "gpt-4.1-mini",
    timeout_secs: float = 12.0,
) -> Optional[callable]:
    """Return a Pipecat handler that runs OpenAI Responses with web_search.

    Returns ``None`` if no OpenAI API key is configured, so the tool can be
    omitted from the schema at registration time.
    """
    if not os.getenv("OPENAI_API_KEY"):
        return None

    try:
        from openai import OpenAI

        client = OpenAI()
    except Exception as e:
        logger.warning(f"OpenAI client init failed: {e}; web_search disabled")
        return None

    async def web_search(params: FunctionCallParams) -> None:
        query = (params.arguments.get("query") or "").strip()
        if not query:
            await params.result_callback({"error": "query is required"})
            return

        logger.info(f"🌐 web_search: {query!r}")

        def _run() -> dict:
            resp = client.responses.create(
                model=model,
                tools=[{"type": "web_search_preview"}],
                input=query,
            )
            # Extract the synthesized answer text.
            text = getattr(resp, "output_text", "") or ""
            # Collect citations if the model attached any.
            citations: list[dict] = []
            for item in getattr(resp, "output", []) or []:
                for c in getattr(item, "content", []) or []:
                    for ann in getattr(c, "annotations", None) or []:
                        if getattr(ann, "type", "") == "url_citation":
                            citations.append(
                                {
                                    "title": getattr(ann, "title", "") or "",
                                    "url": getattr(ann, "url", "") or "",
                                }
                            )
            return {"answer": text.strip(), "citations": citations}

        try:
            result = await asyncio.wait_for(asyncio.to_thread(_run), timeout=timeout_secs)
        except asyncio.TimeoutError:
            await params.result_callback({"error": f"web_search timed out after {timeout_secs}s"})
            return
        except Exception as e:
            logger.error(f"web_search failed: {e}")
            await params.result_callback({"error": str(e)})
            return

        await params.result_callback(result)

    return web_search
