"""Rolling buffer of recent room transcripts.

In LISTEN mode every transcribed user turn lands here, regardless of
whether it triggered the LLM. When a *triggered* turn fires, the recent
buffer is dumped into the LLM context so Alex can answer questions that
reference what was said earlier in the meeting.

Defaults to a 5-minute window. Memory cost at meeting-rate (~30 turns/min
of room talk × ~100 chars/turn) is well under 1 MB even for an all-day
session, so we don't aggressively bound it by entry count.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass


@dataclass
class TurnEntry:
    text: str
    ts: float          # unix time
    triggered: bool    # did this turn fire the LLM (i.e. mentioned a trigger phrase)


class RollingTranscriptBuffer:
    def __init__(self, window_minutes: float = 5.0) -> None:
        self.window_secs = window_minutes * 60
        self._entries: deque[TurnEntry] = deque()

    def add(self, text: str, *, triggered: bool) -> None:
        text = (text or "").strip()
        if not text:
            return
        self._entries.append(TurnEntry(text=text, ts=time.time(), triggered=triggered))
        self._evict()

    def _evict(self) -> None:
        cutoff = time.time() - self.window_secs
        while self._entries and self._entries[0].ts < cutoff:
            self._entries.popleft()

    def recent(self) -> list[TurnEntry]:
        self._evict()
        return list(self._entries)

    def render_for_llm(self, *, max_chars: int = 4000) -> str:
        """Compact multi-line dump for injection into a system / user message."""
        self._evict()
        if not self._entries:
            return ""
        lines: list[str] = []
        # Walk newest → oldest so the model sees the most relevant context
        # even if we hit max_chars and have to drop older lines.
        chars = 0
        for entry in reversed(self._entries):
            mark = "→" if entry.triggered else " "
            line = f"{mark} {entry.text}"
            if chars + len(line) > max_chars:
                break
            lines.append(line)
            chars += len(line)
        return "\n".join(reversed(lines))
