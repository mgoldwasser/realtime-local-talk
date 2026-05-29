# In-room audio setup

This guide walks through wiring Alex into a real conference room so that:

1. Alex hears the room (via a USB conference speakerphone)
2. People in the room hear Alex (through the same speakerphone)
3. Remote Zoom participants also hear Alex (via a virtual mic that BlackHole feeds into Zoom)

## Hardware (recommended)

- **Jabra Speak2 75** (preferred) or **Poly Sync 60** — USB conference speakerphone with hardware AEC and Zoom certification. Either is plug-and-play on macOS.

Cheaper fallback for desk testing: a wired headset (skips the echo problem entirely, but isn't useful in a real conference room).

## Software

### 1. Install BlackHole 2ch

```bash
brew install blackhole-2ch
```

The installer needs your sudo password; if `brew install` fails in a non-interactive shell, install the `.pkg` from https://existential.audio/blackhole/ directly. **Reboot** after install (required for the kernel extension).

### 2. Verify the device appears

```bash
uv run python -m alex.audio.routing
```

You should see `BlackHole 2ch` in the device list along with your built-in mic, speakers, Zoom virtual device, etc.

### 3. Build the routing in `Audio MIDI Setup`

Open `Audio MIDI Setup.app` (in `/Applications/Utilities/`):

**Multi-Output Device** (system output → both room speakers AND BlackHole):
1. Click `+` (bottom-left) → "Create Multi-Output Device"
2. Check both `Jabra Speak2 75` (or your speakerphone) and `BlackHole 2ch`
3. Rename it to e.g. "Alex Output"
4. Right-click → "Use This Device For Sound Output"

**Aggregate Device** (mic input):
1. Click `+` → "Create Aggregate Device"
2. Check `Jabra Speak2 75` only
3. Rename to e.g. "Alex Input"

### 4. Configure Zoom

- **Microphone**: `BlackHole 2ch` (so Zoom hears Alex)
- **Speaker**: "Alex Output" (the Multi-Output device, so Zoom remote audio also plays through the speakerphone)

The system output is "Alex Output", so when Alex's Polly TTS plays it goes to both the room speakerphone (people in the room hear) and BlackHole (which Zoom uses as mic, so remote participants hear).

### 5. Tell Alex which devices to use

Add to your `.env` (substrings, case-insensitive):

```
ALEX_INPUT_DEVICE=Alex Input
ALEX_OUTPUT_DEVICE=Alex Output
```

Verify with:

```bash
uv run python -c "
from alex.audio.routing import find_device
print('input :', find_device('Alex Input',  kind='input'))
print('output:', find_device('Alex Output', kind='output'))
"
```

### 6. The remaining problem: self-echo

Without Phase 7b AEC, Alex hears Polly through the speakerphone and re-transcribes it. The current workaround in the code is `AlwaysUserMuteStrategy` (mute the mic whenever the bot is speaking). That works but disables barge-in.

The Jabra Speak2 75 has hardware AEC on the speakerphone side which removes most of the echo before it reaches the OS, so combined with the software mute the room test works well. Phase 7b adds WebRTC AEC3 on top so we can re-enable barge-in.

## Quick health check

```bash
uv run python -c "
from alex.audio.routing import setup_check
import json; print(json.dumps(setup_check(), indent=2))
"
```

Look for:
- `blackhole_present: true`
- `speakerphones: ['Jabra Speak2 75']` (or your model)
- `multi_output_device_present: true`
- `aggregate_device_present: true`
