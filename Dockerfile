# Use RunPod's pre-cached pytorch base image
# These layers are already on RunPod worker nodes = near-instant pull
FROM runpod/pytorch:1.0.3-cu1281-torch260-ubuntu2204

# Install cuDNN 8 compat lib (pyannote VAD / onnxruntime needs libcudnn_ops_infer.so.8)
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends libcudnn8 ffmpeg && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

WORKDIR /

# Cache directories
RUN mkdir -p /cache/models /root/.cache/torch /models/faster-whisper-large-v3

# Requirements file
COPY builder/requirements.txt /builder/requirements.txt

# Install dependencies (torch already in base image, just add torchaudio)
RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install "numpy<2.0" \
 && python3 -m pip install torchaudio \
 && python3 -m pip install hf_transfer \
 && python3 -m pip install --no-cache-dir -r /builder/requirements.txt

# Patch whisperx diarize.py: replace use_auth_token with token (pyannote 3.3.2 compat)
RUN sed -i 's/use_auth_token=use_auth_token/token=use_auth_token/g' \
    $(python3 -c "import whisperx; import os; print(os.path.join(os.path.dirname(whisperx.__file__), 'diarize.py'))") \
 && echo "Patched whisperx diarize.py"

# Local VAD model (small, ~18MB)
COPY models/whisperx-vad-segmentation.bin /root/.cache/torch/whisperx-vad-segmentation.bin

# Startup model downloader
COPY builder/download_models_startup.py /builder/download_models_startup.py

# Application code
COPY src .

CMD ["python3", "-u", "/rp_handler.py"]
