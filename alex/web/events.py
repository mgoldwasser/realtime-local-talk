"""In-process pub/sub bus that streams pipeline activity to the web UI.

The pipeline (and its sub-processors) push ``UIEvent``s here; the FastAPI
WebSocket endpoint subscribes and fans them out to every connected
browser. Decoupled so CLI mode can run with no bus at all — every emit
site checks ``if bus is not None``.

Event types (the ``type`` field) the dashboard understands:
    state        {state: "idle"|"listening"|"thinking"|"searching"|"speaking"}
    transcript   {kind: "passive"|"triggered"|"follow_up", text, speaker?}
    tool         {name, detail}
    citation     {title, section, excerpt, score}
    said         {text}
    latency      {turn_id, durations: {...}}
    barge_in     {text, overlap}
    echo         {text, overlap}
    notice       {level: "info"|"warn", text}
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from pipecat.frames.frames import Frame
from pipecat.turns.user_mute import BaseUserMuteStrategy


@dataclass
class UIEvent:
    type: str
    data: dict[str, Any]
    ts: float = field(default_factory=time.time)

    def to_json(self) -> dict:
        return {"type": self.type, "ts": self.ts, **self.data}


class UIEventBus:
    """Fan-out bus. Each subscriber gets its own bounded queue so a slow
    browser tab can't block the pipeline — if a queue fills, we drop the
    oldest event for that subscriber.

    Also carries dashboard control state (e.g. ``muted``) so the pipeline's
    mute strategy can read what the browser set, without a second channel.
    """

    def __init__(self, per_client_maxsize: int = 1000) -> None:
        self._subscribers: set[asyncio.Queue] = set()
        self._maxsize = per_client_maxsize
        # Control state the dashboard toggles and the pipeline reads.
        self.control = {"muted": False}
        # Keep a small history so a browser that connects mid-meeting sees
        # recent context instead of a blank screen.
        self._history: list[UIEvent] = []
        self._history_max = 200

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self._maxsize)
        # Seed with recent history.
        for ev in self._history[-self._history_max :]:
            try:
                q.put_nowait(ev)
            except asyncio.QueueFull:
                break
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    def emit(self, type: str, **data: Any) -> None:
        """Synchronous, non-blocking emit — safe to call from anywhere in
        the pipeline without awaiting."""
        ev = UIEvent(type=type, data=data)
        self._history.append(ev)
        if len(self._history) > self._history_max * 2:
            self._history = self._history[-self._history_max :]
        dead: list[asyncio.Queue] = []
        for q in self._subscribers:
            try:
                q.put_nowait(ev)
            except asyncio.QueueFull:
                # Drop the oldest, then enqueue the newest.
                try:
                    q.get_nowait()
                    q.put_nowait(ev)
                except Exception:
                    dead.append(q)
        for q in dead:
            self._subscribers.discard(q)


class DashboardMuteStrategy(BaseUserMuteStrategy):
    """Mutes the mic whenever the dashboard's mute toggle is on.

    Reads ``bus.control['muted']`` per frame, so the browser button takes
    effect immediately without restarting the pipeline. Composes with the
    other mute strategies (OR-combined — muted if any says mute)."""

    def __init__(self, bus: "UIEventBus") -> None:
        super().__init__()
        self._bus = bus

    async def process_frame(self, frame: Frame) -> bool:  # noqa: ARG002
        return bool(self._bus.control.get("muted"))
