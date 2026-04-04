FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

# Install cuDNN 8 compat lib BEFORE removing NVIDIA repos
# (pyannote VAD / onnxruntime needs libcudnn_ops_infer.so.8)
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends libcudnn8 && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN rm -f /etc/apt/sources.list.d/*.list

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash

WORKDIR /

# Update and upgrade the system packages
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install --yes --no-install-recommends sudo ca-certificates git wget curl bash libgl1 libx11-6 software-properties-common ffmpeg build-essential -y &&\
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Install Python 3.10
RUN apt-get update -y && \
    apt-get install python3.10 python3.10-dev python3.10-venv python3-pip -y --no-install-recommends && \
    ln -s /usr/bin/python3.10 /usr/bin/python && \
    rm -f /usr/bin/python3 && \
    ln -s /usr/bin/python3.10 /usr/bin/python3 && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Cache directories
RUN mkdir -p /cache/models /root/.cache/torch /models/faster-whisper-large-v3

# Requirements file
COPY builder/requirements.txt /builder/requirements.txt

# Install torch+torchaudio from cu121 ONLY (no PyPI fallback)
RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install "numpy<2.0" \
 && python3 -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies from PyPI
RUN python3 -m pip install hf_transfer \
 && python3 -m pip install --no-cache-dir -r /builder/requirements.txt

# Patch whisperx diarize.py: replace use_auth_token with token (pyannote 3.3.2 compat)
RUN sed -i 's/use_auth_token=use_auth_token/token=use_auth_token/g' /usr/local/lib/python3.10/dist-packages/whisperx/diarize.py \
 && echo "Patched whisperx diarize.py: use_auth_token -> token"

# Local VAD model (small, ~18MB)
COPY models/whisperx-vad-segmentation.bin /root/.cache/torch/whisperx-vad-segmentation.bin

# ── SLIM IMAGE: Models download at startup instead of bake-in ──
COPY builder/download_models_startup.py /builder/download_models_startup.py

# Application code
COPY src .

CMD ["python3", "-u", "/rp_handler.py"]
