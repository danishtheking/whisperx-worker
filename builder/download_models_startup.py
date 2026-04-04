"""Download all required models at worker startup if not already present.
Runs once per cold start. Models are cached in the container disk."""
import os
import sys
import time
import logging

logger = logging.getLogger("model_downloader")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(handler)


def download_whisper_model(model_dir="/models/faster-whisper-large-v3"):
    """Download Faster Whisper large-v3 model if not present."""
    model_bin = os.path.join(model_dir, "model.bin")
    if os.path.exists(model_bin):
        logger.info("Whisper model already present, skipping download.")
        return

    logger.info("Downloading Faster Whisper large-v3 model (~3GB)...")
    os.makedirs(model_dir, exist_ok=True)

    import urllib.request
    base_url = "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main"
    files = ["config.json", "model.bin", "preprocessor_config.json", "tokenizer.json", "vocabulary.json"]

    for fname in files:
        dest = os.path.join(model_dir, fname)
        if not os.path.exists(dest):
            logger.info(f"  Downloading {fname}...")
            urllib.request.urlretrieve(f"{base_url}/{fname}", dest)

    logger.info("Whisper model download complete.")


def download_speechbrain_ecapa():
    """Download SpeechBrain ECAPA model if not present."""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--speechbrain--spkrec-ecapa-voxceleb")
    if os.path.exists(cache_dir):
        logger.info("SpeechBrain ECAPA model already present, skipping.")
        return

    logger.info("Downloading SpeechBrain ECAPA model...")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id="speechbrain/spkrec-ecapa-voxceleb")
    logger.info("SpeechBrain ECAPA model download complete.")


def download_pyannote_models():
    """Download gated PyAnnote models if not present."""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--pyannote--speaker-diarization-3.1")
    if os.path.exists(cache_dir):
        logger.info("PyAnnote models already present, skipping.")
        return

    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        logger.warning("No HF_TOKEN set — cannot download gated pyannote models. Diarization will fail.")
        return

    logger.info("Downloading PyAnnote models (gated, needs HF token)...")
    from huggingface_hub import snapshot_download

    models = [
        "pyannote/embedding",
        "pyannote/segmentation",
        "pyannote/segmentation-3.0",
        "pyannote/speaker-diarization-2.1",
        "pyannote/speaker-diarization-3.1",
    ]
    for model_id in models:
        logger.info(f"  Downloading {model_id}...")
        snapshot_download(repo_id=model_id, token=hf_token)

    logger.info("PyAnnote models download complete.")


def ensure_vad_model():
    """Ensure VAD segmentation model is in the right place."""
    dest = os.path.expanduser("~/.cache/torch/whisperx-vad-segmentation.bin")
    src = "/root/.cache/torch/whisperx-vad-segmentation.bin"
    if os.path.exists(dest) or os.path.exists(src):
        logger.info("VAD model already present.")
        return
    logger.warning("VAD model not found — whisperx will download it automatically.")


def download_all_models():
    """Download all required models. Called at worker startup."""
    start = time.time()
    logger.info("=== Checking/downloading models ===")

    download_whisper_model()
    download_speechbrain_ecapa()
    download_pyannote_models()
    ensure_vad_model()

    elapsed = time.time() - start
    logger.info(f"=== Model check complete in {elapsed:.1f}s ===")


if __name__ == "__main__":
    download_all_models()
