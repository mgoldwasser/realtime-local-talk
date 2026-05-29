# Linux container for training a custom openWakeWord "Hey Alex" model.
# openWakeWord's training pipeline doesn't run on macOS (requires Piper TTS
# binaries and a tensorflow-lite stack that's awkward on Apple Silicon), so
# we do it in Linux x86_64 via Docker.
#
# Build:
#   docker build -t alex-wakeword-trainer alex/wake/trainer/
#
# Run (mounts the output dir so the .onnx lands on the host):
#   docker run --rm -v "$(pwd)/models:/out" alex-wakeword-trainer

FROM --platform=linux/amd64 python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# openWakeWord training pulls a lot of heavy ML deps; pin versions to a
# known-good combination per their docs.
RUN pip install --no-cache-dir \
    openwakeword[full]==0.4.0 \
    onnx==1.20.0 \
    onnxruntime==1.23.0 \
    piper-phonemize==1.1.0 \
    tensorflow-cpu==2.18.0 \
    torch==2.5.1 \
    torchaudio==2.5.1 \
    datasets==3.1.0 \
    pyyaml==6.0.2 \
    librosa==0.10.2 \
    audiomentations==0.37.0 \
    pydub==0.25.1

WORKDIR /work
COPY train_hey_alex.py recipe.yml ./

ENV WAKEWORD_OUT=/out
CMD ["python", "train_hey_alex.py"]
