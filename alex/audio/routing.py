"""CoreAudio device discovery and selection helpers.

The plan's in-room topology is:

    Multi-Output Device:    Speak2 75 + BlackHole 2ch         (system output)
    Aggregate Device:       Speak2 75                          (mic input)

When Alex speaks (Polly TTS), audio goes to the Multi-Output → both the
room speakerphone (room hears) and BlackHole. BlackHole feeds Zoom's mic
input, so remote participants hear Alex too. The Aggregate Device gives
Alex its own clean mic stream.

This module helps build / verify that topology from Python:
- ``list_devices()``   — pretty-print everything PortAudio sees
- ``find_device()``     — substring-match a device by name
- ``setup_check()``     — verify BlackHole is present + flag missing pieces
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import sounddevice as sd
from rich.console import Console
from rich.table import Table

BLACKHOLE_NAME_HINTS = ("BlackHole", "blackhole")


@dataclass
class Device:
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float

    def is_input(self) -> bool:
        return self.max_input_channels > 0

    def is_output(self) -> bool:
        return self.max_output_channels > 0


def enumerate_devices() -> list[Device]:
    return [
        Device(
            index=i,
            name=d["name"],
            max_input_channels=d["max_input_channels"],
            max_output_channels=d["max_output_channels"],
            default_samplerate=d["default_samplerate"],
        )
        for i, d in enumerate(sd.query_devices())
    ]


def find_device(name_substring: str, *, kind: str = "input") -> Optional[Device]:
    for d in enumerate_devices():
        if (kind == "input" and not d.is_input()) or (kind == "output" and not d.is_output()):
            continue
        if name_substring.lower() in d.name.lower():
            return d
    return None


def list_devices() -> None:
    """Pretty-print all visible audio devices."""
    table = Table(title="CoreAudio devices (via PortAudio)")
    table.add_column("#", justify="right", style="cyan")
    table.add_column("in", justify="center")
    table.add_column("out", justify="center")
    table.add_column("sr", justify="right")
    table.add_column("name")
    for d in enumerate_devices():
        table.add_row(
            str(d.index),
            "✓" if d.is_input() else "",
            "✓" if d.is_output() else "",
            f"{d.default_samplerate:.0f}",
            d.name,
        )
    Console().print(table)


def setup_check() -> dict:
    """Return a small status dict the setup guide can render."""
    devices = enumerate_devices()
    blackhole = next(
        (d for d in devices if any(h in d.name for h in BLACKHOLE_NAME_HINTS)), None
    )
    speakerphones = [
        d for d in devices
        if any(k in d.name.lower() for k in ("jabra", "speak2", "poly sync", "sync 60"))
    ]
    multi_out = next((d for d in devices if "multi-output" in d.name.lower()), None)
    aggregate = next((d for d in devices if "aggregate" in d.name.lower()), None)
    return {
        "blackhole_present": blackhole is not None,
        "blackhole": blackhole.name if blackhole else None,
        "speakerphones": [d.name for d in speakerphones],
        "multi_output_device_present": multi_out is not None,
        "aggregate_device_present": aggregate is not None,
    }


if __name__ == "__main__":
    list_devices()
    print()
    print("Setup check:", setup_check())
