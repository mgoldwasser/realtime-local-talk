# Acoustic Echo Cancellation (AEC) — current status

## The problem

When Alex's TTS plays through speakers (laptop or speakerphone), the mic picks it up, Whisper transcribes it, and the LLM gets called on "what Alex just said" — an echo loop. We saw this in early Phase 2 voice testing.

## Current mitigation: software self-mute

`AlwaysUserMuteStrategy` (Pipecat built-in) gates the user input frames whenever the bot is speaking. This is wired by default in voice modes. Trade-off: no barge-in (you can't interrupt Alex by speaking — you have to wait or Ctrl-C).

For most demos this is fine. The whole point of the in-room mode is committee Q&A, not rapid back-and-forth.

## The proper fix (Phase 7b proper): WebRTC AEC3

The plan called for WebRTC AEC3 wired on the mic stream with the TTS PCM buffer as the reference signal. This would remove Alex's voice from the mic stream so barge-in works without echo.

**Status: blocked on Apple Silicon.** All three Python packages we tried (`webrtc-audio-processing`, `webrtc-audio-processing-2`, `pyspeexdsp`/`speexdsp-python`) fail to build on arm64 macOS due to hardcoded x86 flags like `-mfpu=`. This is a known ecosystem hole. Options for future work:

1. **Build webrtc-apm from source with arm64 patches.** ~half day of engineering. Doable.
2. **Krisp SDK** — commercial, has macOS arm64 support, would be a clean drop-in via a Pipecat `BaseAudioFilter`. ~$30/agent/month commercial pricing; not in our current vendor allowlist.
3. **System-level AEC via `kAudioUnitSubType_VoiceProcessingIO`** — macOS has built-in AEC in this AudioUnit but PortAudio doesn't expose it. Would need a custom CoreAudio wrapper. Significant engineering.

## Recommended path for v1

Use a USB conference speakerphone with hardware AEC — the **Jabra Speak2 75** or **Poly Sync 60**. Both apply DSP AEC before the audio ever leaves the device, so the mic stream macOS sees has Alex's voice mostly removed.

With hardware AEC present, set `ALEX_HARDWARE_AEC_PRESENT=true` in `.env`. This disables `AlwaysUserMuteStrategy` so barge-in works. Software AEC stays a future improvement.

## Quick reference

| Configuration | Echo handling | Barge-in |
|---|---|---|
| Default (laptop mic + speakers) | Software self-mute | No |
| Hardware AEC (Speak2 75) + `ALEX_HARDWARE_AEC_PRESENT=true` | Hardware AEC handles it | Yes |
| Future: WebRTC AEC3 software | Software + hardware layered | Yes |
| Future: Krisp SDK | Commercial AEC | Yes |
