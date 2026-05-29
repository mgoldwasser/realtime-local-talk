"""Global push-to-talk hotkey.

Tap-to-toggle semantics (not hold-to-talk): one press starts listening,
another stops. Designed for a committee setting where the operator isn't
going to hold a key down for the whole meeting.

Default hotkey: ``<alt>+<space>`` (Option+Space on macOS). Customize via
``hotkey`` arg or ``ALEX_PTT_HOTKEY`` env (uses pynput's HotKey grammar).

macOS note: needs Accessibility permission on the *terminal* running this
process (System Settings → Privacy & Security → Accessibility). pynput
fails silently without it; we log a hint on startup.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Optional

from loguru import logger
from pipecat.frames.frames import Frame
from pipecat.turns.user_mute import BaseUserMuteStrategy


@dataclass
class PTTState:
    """Shared toggle the hotkey thread flips and the mute strategy reads."""

    listening: bool = False
    on_toggle: Optional[Callable[[bool], None]] = None

    def toggle(self) -> None:
        self.listening = not self.listening
        if self.on_toggle:
            self.on_toggle(self.listening)


class PTTMuteStrategy(BaseUserMuteStrategy):
    """Mute everything except while a PTT toggle is active."""

    def __init__(self, state: PTTState) -> None:
        super().__init__()
        self._state = state

    async def process_frame(self, frame: Frame) -> bool:  # noqa: ARG002
        return not self._state.listening


class PTTHotkeyListener:
    """Background thread that listens for a global hotkey and toggles ``state``.

    Daemon thread so the process can exit cleanly without waiting on it.
    """

    def __init__(self, state: PTTState, hotkey: str = "<alt>+<space>") -> None:
        self._state = state
        self._hotkey = hotkey
        self._listener = None  # pynput keyboard.GlobalHotKeys

    def start(self) -> None:
        try:
            from pynput import keyboard
        except Exception as e:
            logger.warning(f"PTT hotkey disabled: pynput unavailable ({e})")
            return

        def _on_activate() -> None:
            self._state.toggle()

        try:
            self._listener = keyboard.GlobalHotKeys({self._hotkey: _on_activate})
            self._listener.daemon = True
            self._listener.start()
            logger.info(
                f"PTT hotkey active: {self._hotkey} (tap to start/stop listening)"
            )
            logger.info(
                "if it doesn't fire: grant Accessibility permission to your "
                "terminal in System Settings → Privacy & Security"
            )
        except Exception as e:
            logger.warning(f"PTT hotkey listener failed to start: {e}")

    def stop(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
