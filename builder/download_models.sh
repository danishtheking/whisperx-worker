#!/bin/bash
set -e

CACHE_DIR="/cache/models"
MODELS_DIR="/models"

mkdir -p /root/.cache/torch/hub/checkpoints

download() {
  local file_url="$1"
  local destination_path="$2"
  local cache_path="${CACHE_DIR}/${destination_path##*/}"

  mkdir -p "$(dirname "$cache_path")"
  mkdir -p "$(dirname "$destination_path")"

  if [ ! -e "$cache_path" ]; then
    echo "Downloading $file_url to cache..."
    wget -O "$cache_path" "$file_url"
  else
    echo "Using cached version of ${cache_path##*/}"
  fi

  cp "$cache_path" "$destination_path"
}

# Download Faster Whisper Model
faster_whisper_model_dir="${MODELS_DIR}/faster-whisper-large-v3"
mkdir -p "$faster_whisper_model_dir"

download "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/config.json"              "$faster_whisper_model_dir/config.json"
download "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/model.bin"              "$faster_whisper_model_dir/model.bin"
download "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/preprocessor_config.json" "$faster_whisper_model_dir/preprocessor_config.json"
download "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/tokenizer.json"          "$faster_whisper_model_dir/tokenizer.json"
download "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/vocabulary.json"         "$faster_whisper_model_dir/vocabulary.json"

echo "Faster Whisper model downloaded."

# Python block: HF downloads using secret — use 'token' not 'use_auth_token'
python3 -c "
import os

hf_token = None
try:
    with open('/run/secrets/hf_token', 'r') as f:
        hf_token = f.read().strip()
        print(f'Read HF token from secret file: {hf_token[:10]}...')
except Exception as e:
    print(f'No secret file found: {e}')
    hf_token = os.environ.get('HF_TOKEN')
    if hf_token:
        print(f'Using HF_TOKEN from env: {hf_token[:10]}...')

from huggingface_hub import snapshot_download

# Download SpeechBrain (public, no token needed)
snapshot_download(repo_id='speechbrain/spkrec-ecapa-voxceleb')
print('SpeechBrain ECAPA downloaded.')

# Download PyAnnote models (gated, need token)
if hf_token:
    snapshot_download(repo_id='pyannote/embedding', token=hf_token)
    print('pyannote/embedding downloaded.')
    snapshot_download(repo_id='pyannote/speaker-diarization-2.1', token=hf_token)
    print('pyannote/speaker-diarization-2.1 downloaded.')
else:
    print('WARNING: No HF_TOKEN — skipping pyannote model downloads')
"

echo "All models downloaded successfully."
