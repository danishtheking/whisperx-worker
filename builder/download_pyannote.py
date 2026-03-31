"""Download gated PyAnnote models using HF token from BuildKit secret or env."""
import os
from huggingface_hub import snapshot_download

# Read HF token from BuildKit secret or environment
hf_token = None
try:
    with open('/run/secrets/hf_token', 'r') as f:
        hf_token = f.read().strip()
        print(f'Using HF token from secret: {hf_token[:10]}...')
except Exception as e:
    print(f'No secret file found: {e}')
    hf_token = os.environ.get('HF_TOKEN')
    if hf_token:
        print(f'Using HF_TOKEN from env: {hf_token[:10]}...')

if not hf_token:
    raise RuntimeError('No HF token available — cannot download gated pyannote models.')

snapshot_download(repo_id='pyannote/embedding', token=hf_token)
print('pyannote/embedding downloaded')

snapshot_download(repo_id='pyannote/speaker-diarization-2.1', token=hf_token)
print('pyannote/speaker-diarization-2.1 downloaded')
