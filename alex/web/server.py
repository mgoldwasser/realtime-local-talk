"""FastAPI dashboard server for Alex.

Architecture (observer dashboard, audio stays local):

    uvicorn ──serves──> static/index.html  (the dashboard)
        │
        ├── GET  /              → dashboard page
        ├── WS   /ws            → streams UIEvents to the browser;
        │                         receives control messages back
        └── on startup: launch the Alex pipeline as a background task,
            with a UIEventBus wired in. The pipeline uses the normal
            LocalAudioTransport — mic + speakers stay on the Mac. The
            browser is a read-only-ish window into what Alex is doing,
            plus a few controls (mute, pause).

Run:
    uv run alex-web              # then open http://localhost:8765
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

import typer
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from alex.config import Activation, get_settings
from alex.web.events import UIEventBus

STATIC_DIR = Path(__file__).parent / "static"

# Shared across the app: one bus, one pipeline task. The bus also carries
# dashboard control state (bus.control) that the pipeline's mute strategy reads.
_bus = UIEventBus()
_pipeline_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline_task
    load_dotenv()
    settings = get_settings()
    # Web mode forces LISTEN activation — that's the meeting use case.
    settings.activation = Activation.LISTEN

    from alex.pipeline import run_pipeline

    async def _run():
        try:
            await run_pipeline(
                settings=settings,
                text_input=False,
                forced_tier=None,
                silent=False,
                bus=_bus,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.exception(f"pipeline crashed: {e}")
            _bus.emit("notice", level="warn", text=f"pipeline error: {e}")

    _pipeline_task = asyncio.create_task(_run())
    logger.info("Alex pipeline started; dashboard at http://localhost:8765")
    yield
    if _pipeline_task:
        _pipeline_task.cancel()
        try:
            await _pipeline_task
        except (asyncio.CancelledError, Exception):
            pass


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()
    queue = _bus.subscribe()

    async def _pump_out():
        """Bus → browser."""
        try:
            while True:
                ev = await queue.get()
                await websocket.send_text(json.dumps(ev.to_json()))
        except (WebSocketDisconnect, RuntimeError):
            pass

    async def _pump_in():
        """Browser → control state."""
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                action = msg.get("action")
                if action == "mute":
                    _bus.control["muted"] = bool(msg.get("value", True))
                    _bus.emit(
                        "notice",
                        level="info",
                        text=f"mic {'muted' if _bus.control['muted'] else 'live'} (from dashboard)",
                    )
                # Future: pause, change-trigger, force-summary, etc.
        except (WebSocketDisconnect, RuntimeError):
            pass

    out = asyncio.create_task(_pump_out())
    inp = asyncio.create_task(_pump_in())
    try:
        await asyncio.wait({out, inp}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        out.cancel()
        inp.cancel()
        _bus.unsubscribe(queue)


# Static assets (if we add CSS/JS files later).
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


cli = typer.Typer(no_args_is_help=False, add_completion=False)


@cli.command()
def main(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8765, "--port"),
) -> None:
    """Launch the Alex dashboard + pipeline."""
    uvicorn.run(app, host=host, port=port, log_level="warning")


def run() -> None:
    cli()


if __name__ == "__main__":
    run()
