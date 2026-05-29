from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Activation(str, Enum):
    PTT = "ptt"
    WAKEWORD = "wakeword"
    BOTH = "both"


class TtsBackend(str, Enum):
    POLLY = "polly"
    KOKORO = "kokoro"


class LlmTier(str, Enum):
    HAIKU = "haiku"
    SONNET = "sonnet"
    GPT5 = "gpt5"


class Settings(BaseSettings):
    """Runtime configuration. Values come from environment / .env, then CLI flags
    override on top via `Settings(...)` at startup."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="ALEX_",
        extra="ignore",
    )

    # AWS
    aws_region: str = "us-east-1"
    aws_profile: Optional[str] = None

    # Activation
    activation: Activation = Activation.PTT

    # TTS
    tts: TtsBackend = TtsBackend.POLLY
    polly_voice: str = "Ruth"
    polly_engine: str = "generative"

    # LLM (cross-region inference profile IDs from Bedrock console)
    llm_haiku: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    llm_sonnet: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    llm_filler: str = "us.amazon.nova-micro-v1:0"  # cheap stall-talk model

    # Bedrock latency configuration. "standard" works everywhere; "optimized"
    # only on the model + region combos AWS has rolled it out to (e.g. Haiku
    # 3.5 in us-east-2 historically; Haiku 4.5 in us-east-1 is not yet
    # supported as of build time). Override per-environment via ALEX_LLM_LATENCY.
    llm_latency: str = "standard"

    # Wake word. Default uses openWakeWord's bundled `alexa` model — phonetic
    # neighbor of "Alex", good enough for v1. Train a real "Hey Alex" via
    # alex/wake/trainer/ and point this at the resulting .onnx.
    wakeword_model_path: Optional[str] = None  # None → bundled alexa model
    wakeword_threshold: float = 0.5
    wakeword_listen_secs: float = 10.0

    # Audio device names (CoreAudio); empty means system default.
    input_device: str = ""
    output_device: str = ""

    # Set to True only when the input device has its own hardware AEC
    # (e.g. Jabra Speak2 75, Poly Sync 60). Disables the software self-mute
    # so barge-in works. Without hardware AEC this creates an echo loop.
    hardware_aec_present: bool = False

    # STT (MLX-Whisper) — values map to alex.stt.mlx_whisper_service.MLXModel.
    # LARGE_V3_TURBO_Q4 is the plan's pick: best quality/latency tradeoff at
    # ~1.5 GB on Apple Silicon Neural Engine.
    stt_model: str = "LARGE_V3_TURBO_Q4"

    # Paths
    project_root: Path = Path(__file__).resolve().parent.parent
    latency_dir: Path = project_root / "latency_runs"
    vocab_path: Path = project_root / "corpora" / "vocab.yaml"
    lance_db_path: Path = project_root / "data" / "lance"
    lance_table: str = "chunks"
    duck_db_path: Path = project_root / "data" / "duck.db"
    entities_path: Path = project_root / "corpora" / "entities.json"

    def aws_session_kwargs(self) -> dict:
        kw = {"region_name": self.aws_region}
        if self.aws_profile:
            kw["profile_name"] = self.aws_profile
        return kw


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.latency_dir.mkdir(parents=True, exist_ok=True)
    return _settings
