"""WhisperX RunPod Serverless Handler — fixed for NumPy 2.x, lazy model loading."""
import sys
import os
import logging
import math
import copy
import shutil
import json

# ── 0) Logging FIRST so we can see any crash ──────────────────────────
logger = logging.getLogger("rp_handler")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_formatter = logging.Formatter(
    "%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(console_formatter)

file_handler = logging.FileHandler("container_log.txt", mode="a")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    "%(asctime)s %(levelname)s [%(name)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(file_formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.info("=== rp_handler.py starting ===")

# ── 0b) Download models if not already present (slim image) ─────────
try:
    sys.path.insert(0, "/builder")
    from download_models_startup import download_all_models
    download_all_models()
except Exception as e:
    logger.error(f"Model download failed: {e}", exc_info=True)
    # Continue anyway — models might already be cached

# ── 1) Environment & HF auth ─────────────────────────────────────────
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import torch
import numpy as np
from typing import Optional, Any

raw_token = os.environ.get("HF_TOKEN", "")
hf_token = raw_token.strip()

if hf_token:
    try:
        from huggingface_hub import login, whoami
        logger.debug(f"HF_TOKEN Loaded: {repr(hf_token[:10])}...")
        login(token=hf_token, add_to_git_credential=False)
        user = whoami(token=hf_token)
        logger.info(f"Hugging Face Authenticated as: {user['name']}")
    except Exception as e:
        logger.error("Failed to authenticate with Hugging Face", exc_info=True)
else:
    logger.warning("No Hugging Face token found in HF_TOKEN environment variable.")

# ── 2) Import project modules (after logging is set up) ──────────────
try:
    from speaker_profiles import load_embeddings, relabel
    logger.info("speaker_profiles imported OK")
except Exception as e:
    logger.error(f"Failed to import speaker_profiles: {e}", exc_info=True)
    load_embeddings = None
    relabel = None

try:
    from speaker_processing import (
        process_diarized_output, enroll_profiles,
        identify_speakers_on_segments, load_known_speakers_from_samples,
        identify_speaker, relabel_speakers_by_avg_similarity,
    )
    logger.info("speaker_processing imported OK")
except Exception as e:
    logger.error(f"Failed to import speaker_processing: {e}", exc_info=True)

# ── 3) RunPod + predictor ────────────────────────────────────────────
import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils import download_files_from_urls, rp_cleanup
from rp_schema import INPUT_VALIDATIONS
from predict import Predictor, Output

logger.info("Loading WhisperX model...")
MODEL = Predictor()
MODEL.setup()
logger.info("WhisperX model ready")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Helpers ──────────────────────────────────────────────────────────

def _write_base64_audio(job_id: str, audio_b64: str, filename: Optional[str]) -> str:
    import base64
    safe_name = (filename or "audio").replace("/", "_").replace("\\", "_")
    if "." not in safe_name:
        safe_name += ".bin"
    job_dir = os.path.join("/jobs", job_id)
    os.makedirs(job_dir, exist_ok=True)
    out_path = os.path.join(job_dir, safe_name)
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(audio_b64))
    return out_path


def cleanup_job_files(job_id, jobs_directory='/jobs'):
    job_path = os.path.join(jobs_directory, job_id)
    if os.path.exists(job_path):
        try:
            shutil.rmtree(job_path)
            logger.info(f"Removed job directory: {job_path}")
        except Exception as e:
            logger.error(f"Error removing job directory {job_path}: {str(e)}", exc_info=True)


def _to_jsonable(obj: Any):
    """Recursively convert an object to JSON-safe types."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return _to_jsonable(obj.item())
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return round(obj, 6)  # limit decimal precision
    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    try:
        return str(obj)
    except Exception:
        return None


def _has_devanagari(text):
    """Check if text contains Devanagari script characters."""
    return any('\u0900' <= c <= '\u097F' for c in text)


def _romanize_text(text):
    """Convert Devanagari text to Romanized (Hinglish), preserving English parts."""
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate

        # Split text into Devanagari and non-Devanagari chunks
        result = []
        current_chunk = []
        is_devanagari = False

        for char in text:
            char_is_dev = '\u0900' <= char <= '\u097F'
            if char_is_dev != is_devanagari and current_chunk:
                chunk_text = ''.join(current_chunk)
                if is_devanagari:
                    chunk_text = transliterate(chunk_text, sanscript.DEVANAGARI, sanscript.ITRANS)
                    # Clean up ITRANS artifacts for readability
                    chunk_text = chunk_text.replace('.a', 'a').replace('~N', 'n')
                result.append(chunk_text)
                current_chunk = []
            is_devanagari = char_is_dev
            current_chunk.append(char)

        if current_chunk:
            chunk_text = ''.join(current_chunk)
            if is_devanagari:
                chunk_text = transliterate(chunk_text, sanscript.DEVANAGARI, sanscript.ITRANS)
                chunk_text = chunk_text.replace('.a', 'a').replace('~N', 'n')
            result.append(chunk_text)

        return ''.join(result)
    except Exception as e:
        logger.warning(f"Transliteration failed: {e}")
        return text


def _romanize_segments(segments):
    """Romanize Devanagari text in all segments."""
    romanized = []
    for seg in segments:
        text = seg.get("text", "")
        if _has_devanagari(text):
            seg = {**seg, "text": _romanize_text(text)}
        romanized.append(seg)
    return romanized


def _safe_json_output(obj):
    """Ensure output is 100% JSON-serializable. Last line of defense."""
    try:
        cleaned = _to_jsonable(obj)
        # Verify it can actually be serialized
        json.dumps(cleaned)
        return cleaned
    except (TypeError, ValueError, OverflowError) as e:
        logger.error(f"JSON serialization failed: {e}", exc_info=True)
        # Return minimal safe output
        return {"error": f"Output serialization failed: {str(e)}"}


# ── Main handler ─────────────────────────────────────────────────────

def run(job):
    job_id = job["id"]
    job_input = job["input"]

    # Validate basic schema
    validated = validate(job_input, INPUT_VALIDATIONS)
    if "errors" in validated:
        logger.error(f"Validation errors: {validated['errors']}")
        return _safe_json_output({"error": str(validated["errors"])})

    # 1) Obtain primary audio
    audio_file_path = None
    try:
        audio_b64 = job_input.get("audio_base64")
        audio_fname = job_input.get("audio_filename")
        if audio_b64:
            audio_file_path = _write_base64_audio(job_id, audio_b64, audio_fname)
            logger.debug(f"Audio received as base64 -> {audio_file_path}")
        else:
            audio_file_path = download_files_from_urls(
                job_id, [job_input["audio_file"]]
            )[0]
            logger.debug(f"Audio downloaded -> {audio_file_path}")
    except Exception as e:
        logger.error("Audio acquisition failed", exc_info=True)
        return _safe_json_output({"error": f"audio acquisition: {str(e)}"})

    # 2) Download speaker profiles (optional)
    speaker_profiles = job_input.get("speaker_samples", [])
    embeddings = {}
    output_dict = {}
    if speaker_profiles:
        try:
            embeddings = load_known_speakers_from_samples(
                speaker_profiles,
                huggingface_access_token=hf_token,
            )
            logger.info(f"Enrolled {len(embeddings)} speaker profiles successfully.")
        except Exception as e:
            logger.error("Enrollment failed", exc_info=True)
            output_dict["warning"] = f"Enrollment skipped: {e}"

    # 3) Call WhisperX / VAD / diarization
    predict_input = {
        "audio_file": audio_file_path,
        "language": job_input.get("language"),
        "language_detection_min_prob": job_input.get("language_detection_min_prob", 0),
        "language_detection_max_tries": job_input.get("language_detection_max_tries", 5),
        "initial_prompt": job_input.get("initial_prompt"),
        "batch_size": job_input.get("batch_size", 64),
        "temperature": job_input.get("temperature", 0),
        "vad_onset": job_input.get("vad_onset", 0.50),
        "vad_offset": job_input.get("vad_offset", 0.363),
        "align_output": job_input.get("align_output", False),
        "custom_align_model": job_input.get("custom_align_model"),
        "diarization": job_input.get("diarization", False),
        "huggingface_access_token": job_input.get("huggingface_access_token") or hf_token,
        "min_speakers": job_input.get("min_speakers"),
        "max_speakers": job_input.get("max_speakers"),
        "debug": job_input.get("debug", False),
    }

    try:
        result = MODEL.predict(**predict_input)
    except Exception as e:
        logger.error("WhisperX prediction failed", exc_info=True)
        return _safe_json_output({"error": f"prediction: {str(e)}"})

    output_dict.update({
        "segments": result.segments,
        "detected_language": result.detected_language,
    })

    # Transliterate Devanagari → Romanized (Hinglish) for Hindi/Marathi/etc.
    romanize = job_input.get("romanize", True)  # Default ON
    if romanize and output_dict.get("segments"):
        output_dict["segments"] = _romanize_segments(output_dict["segments"])

    # Sanitize output to valid JSON
    output_dict = _safe_json_output(output_dict)
    logger.info(f"Output segments count: {len(output_dict.get('segments', []))}")

    # 4) Speaker verification (optional)
    if embeddings:
        try:
            segments_with_speakers = identify_speakers_on_segments(
                segments=output_dict["segments"],
                audio_path=audio_file_path,
                enrolled=embeddings,
                threshold=0.1,
            )
            segments_with_final_labels = relabel_speakers_by_avg_similarity(segments_with_speakers)
            output_dict["segments"] = segments_with_final_labels
            logger.info("Speaker identification completed successfully.")
        except Exception as e:
            logger.error("Speaker identification failed", exc_info=True)
            output_dict["warning"] = f"Speaker identification skipped: {e}"
    else:
        logger.info("No enrolled embeddings; skipping speaker identification.")

    # 5) Cleanup
    try:
        rp_cleanup.clean(["input_objects"])
        cleanup_job_files(job_id)
    except Exception as e:
        logger.warning(f"Cleanup issue: {e}", exc_info=True)

    # Final safety check — ensure output is JSON-safe
    output_dict = _safe_json_output(output_dict)
    logger.info(f"Returning output ({len(json.dumps(output_dict))} bytes)")
    return output_dict


runpod.serverless.start({"handler": run})
