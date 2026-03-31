FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

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

# 2. Cache directories
RUN mkdir -p /cache/models /root/.cache/torch

# 4. Requirements file
COPY builder/requirements.txt /builder/requirements.txt

# 5. Python dependencies — pin numpy<2 BEFORE everything else
RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install "numpy<2.0" \
 && python3 -m pip install hf_transfer \
 && python3 -m pip install --no-cache-dir -r /builder/requirements.txt

# 6. Local VAD model
COPY models/whisperx-vad-segmentation.bin /root/.cache/torch/whisperx-vad-segmentation.bin

# 7. Builder scripts + model downloader
COPY builder /builder
RUN chmod +x /builder/download_models.sh
RUN --mount=type=secret,id=hf_token /builder/download_models.sh

# 8. Application code
COPY src .

CMD ["python3", "-u", "/rp_handler.py"]
