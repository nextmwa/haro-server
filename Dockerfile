FROM python:3.11-slim

# espeak-ng: required by Kokoro's phonemizer backend for non-English G2P
# (e.g. Italian), confirmed in Task 6's report against the real `kokoro`
# package. ffmpeg: commonly needed by audio-processing libraries in this
# stack (librosa/torchaudio backends used by nemo_toolkit's STT pipeline).
# Task 7's report on the real `nemo_toolkit` install did not surface any
# further system packages beyond what pip installs itself.
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir -e .

EXPOSE 8765

CMD ["python", "-m", "haro_server.main"]
