import os
import torch
import librosa
import numpy as np
import tempfile
import requests
import logging
import sys
from collections import defaultdict
from datetime import datetime
from scipy.spatial.distance import cosine, cdist

# Lazy-loaded globals
_EMBED_MODEL = None
_ECAPA = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
HF_TOKEN = os.getenv("HF_TOKEN")

# Set up logging
logger = logging.getLogger("speaker_processing")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def _get_embed_model():
    """Lazy-load pyannote embedding model on first use."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from pyannote.audio import Inference
        _EMBED_MODEL = Inference("pyannote/embedding", device=DEVICE, use_auth_token=HF_TOKEN)
        logger.info("Loaded pyannote embedding model")
    return _EMBED_MODEL


def _get_ecapa():
    """Lazy-load SpeechBrain ECAPA model on first use."""
    global _ECAPA
    if _ECAPA is None:
        try:
            from speechbrain.inference import EncoderClassifier
        except ImportError:
            from speechbrain.pretrained import EncoderClassifier
        _ECAPA = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": DEVICE},
        )
        logger.info("Loaded SpeechBrain ECAPA model")
    return _ECAPA


def spk_embed(wave_16k_mono: np.ndarray) -> np.ndarray:
    """Return 192-D embedding for one mono waveform @16 kHz."""
    ecapa = _get_ecapa()
    wav = torch.tensor(wave_16k_mono).unsqueeze(0).to(DEVICE)
    return ecapa.encode_batch(wav).squeeze(0).cpu().numpy()


# ------------------------------------------------------------------
# Helper so we never forget the new 3.x input format
def to_pyannote_dict(wf, sr=16000):
    """Return mapping accepted by pyannote.audio 3.x Inference."""
    if isinstance(wf, np.ndarray):
        wf = torch.tensor(wf, dtype=torch.float32)
    if wf.ndim == 1:
        wf = wf.unsqueeze(0)
    return {"waveform": wf, "sample_rate": sr}


def to_numpy(arr) -> np.ndarray:
    """Return a 1-D numpy embedding whatever pyannote gives back."""
    if isinstance(arr, np.ndarray):
        return arr.flatten()
    if torch.is_tensor(arr):
        return arr.detach().cpu().numpy().flatten()
    from pyannote.core import SlidingWindowFeature
    if isinstance(arr, SlidingWindowFeature):
        return arr.data.flatten()
    raise TypeError(f"Unsupported embedding type: {type(arr)}")


# Global cache for computed speaker embeddings.
_SPEAKER_EMBEDDING_CACHE = {}


def _to_numpy_flat(emb):
    """Return a 1-D numpy array from various embedding types."""
    from pyannote.core import SlidingWindowFeature

    if isinstance(emb, torch.Tensor):
        return emb.detach().cpu().numpy().flatten()
    if isinstance(emb, SlidingWindowFeature):
        return emb.data.flatten()
    data = getattr(emb, "data", None)
    if isinstance(data, np.ndarray):
        return data.flatten()
    raise TypeError(f"Unsupported embedding type: {type(emb)}")


def load_known_speakers_from_samples(speaker_samples, huggingface_access_token=None):
    global _SPEAKER_EMBEDDING_CACHE
    known_embeddings = {}

    try:
        model = _get_embed_model()
    except Exception as e:
        logger.error(f"Failed to load pyannote embedding model: {e}", exc_info=True)
        return {}

    for sample in speaker_samples:
        name = sample.get("name")
        url = sample.get("url")
        if not name:
            if url:
                name = os.path.splitext(os.path.basename(url))[0]
                logger.debug(f"No name provided; using '{name}' from URL.")
            else:
                logger.error(f"Skipping sample with missing name and URL: {sample}")
                continue

        if name in _SPEAKER_EMBEDDING_CACHE:
            logger.debug(f"Using cached embedding for speaker '{name}'.")
            known_embeddings[name] = _SPEAKER_EMBEDDING_CACHE[name]
            continue

        filepath = None
        if sample.get("file_path"):
            filepath = sample["file_path"]
            logger.debug(f"Loading speaker sample '{name}' from local file: {filepath}")
        elif url:
            try:
                logger.debug(f"Downloading speaker sample '{name}' from URL: {url}")
                response = requests.get(url)
                response.raise_for_status()
                suffix = os.path.splitext(url)[1]
                if not suffix:
                    suffix = ".wav"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(response.content)
                    tmp.flush()
                    filepath = tmp.name
            except Exception as e:
                logger.error(f"Failed to download speaker sample '{name}' from {url}: {e}", exc_info=True)
                continue
        else:
            logger.error(f"Skipping sample '{name}': no file_path or URL provided.")
            continue

        try:
            waveform, sr = librosa.load(filepath, sr=16000, mono=True)
            emb = model(to_pyannote_dict(waveform, sr))
            if hasattr(emb, "data"):
                emb_np = np.mean(emb.data, axis=0)
            else:
                emb_np = emb.cpu().numpy() if isinstance(emb, torch.Tensor) else np.asarray(emb)
            emb_np = emb_np / np.linalg.norm(emb_np)
            _SPEAKER_EMBEDDING_CACHE[name] = emb_np
            known_embeddings[name] = emb_np
            logger.debug(f"Computed embedding for '{name}' (norm={np.linalg.norm(emb_np):.2f}).")
        except Exception as e:
            logger.error(f"Failed to process speaker sample '{name}' from file {filepath}: {e}", exc_info=True)

        if not sample.get("file_path") and url and filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception as e:
                logger.warning(f"Could not remove temporary file {filepath}: {e}")

    return known_embeddings


def identify_speaker(segment_embedding, known_embeddings, threshold=0.1):
    if isinstance(segment_embedding, np.ndarray):
        segment_embedding = segment_embedding.ravel()
    else:
        logger.error("Invalid segment_embedding type, expected numpy.ndarray")
        return "Unknown", -1

    best_match, best_similarity = "Unknown", -1.0
    for speaker, known_emb in known_embeddings.items():
        if not isinstance(known_emb, np.ndarray):
            continue
        known_emb_flat = known_emb.ravel()
        score = 1 - cosine(segment_embedding, known_emb_flat)
        if score > best_similarity:
            best_similarity, best_match = score, speaker

    return (best_match, best_similarity) if best_similarity >= threshold else ("Unknown", best_similarity)


def embed_waveform(wav: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Return a 512-dim L2-normalized embedding for a waveform."""
    model = _get_embed_model()
    feat = model({"waveform": torch.tensor(wav).unsqueeze(0), "sample_rate": sr})
    if hasattr(feat, "data"):
        arr = feat.data.mean(axis=0)
    else:
        arr = feat.squeeze(0).cpu().numpy()
    arr = arr.astype(np.float32)
    return arr / np.linalg.norm(arr)


def enroll_profiles(profiles: list) -> dict:
    embeddings = {}
    for p in profiles:
        wav, sr = librosa.load(p["file_path"], sr=16000, mono=True)
        embeddings[p["name"]] = embed_waveform(wav, sr)
    return embeddings


def identify_speakers_on_segments(segments, audio_path, enrolled, threshold=0.1):
    names = list(enrolled.keys())
    mat = np.stack([enrolled[n] for n in names])

    for seg in segments:
        wav, sr = librosa.load(audio_path, sr=16000, mono=True,
                               offset=seg["start"],
                               duration=seg["end"] - seg["start"])
        emb = embed_waveform(wav, sr)
        sims = 1 - cdist(emb[None, :], mat, metric="cosine")[0]
        best = sims.argmax()
        if sims[best] >= threshold:
            seg["speaker_id"] = names[best]
            seg["similarity"] = float(sims[best])
        else:
            seg["speaker_id"] = "Unknown"
            seg["similarity"] = float(sims.max())
    return segments


def relabel_speakers_by_avg_similarity(segments):
    grouped = defaultdict(list)
    for seg in segments:
        spk = seg.get("speaker")
        sim = seg.get("similarity")
        sid = seg.get("speaker_id")
        if spk and sim is not None and sid:
            grouped[spk].append((sid, sim))

    relabel_map = {}
    for orig_spk, samples in grouped.items():
        scores = defaultdict(list)
        for sid, sim in samples:
            scores[sid].append(sim)
        avg = {sid: sum(vals) / len(vals) for sid, vals in scores.items()}
        best_match = max(avg, key=avg.get)
        relabel_map[orig_spk] = best_match

    for seg in segments:
        spk = seg.get("speaker")
        if spk in relabel_map:
            seg["speaker"] = relabel_map[spk]

    return segments


def process_diarized_output(output, audio_filepath, known_embeddings,
                            huggingface_access_token=None, return_logs=True,
                            threshold=0.20):
    log_data = {
        "segments": [],
        "centroids": {},
        "relabeling_decisions": [],
        "timestamp": datetime.now().isoformat(),
    }

    embedder = _get_embed_model()
    segments = output.get("segments", [])
    if not segments:
        return output, None

    # 1) Embed each diarized segment
    for seg in segments:
        seg.setdefault("speaker", "Unknown")
        start, end = seg["start"], seg["end"]
        try:
            wav, _ = librosa.load(audio_filepath, sr=16000, mono=True, offset=start, duration=end - start)
        except Exception as e:
            logger.error(f"Could not load [{start:.2f}-{end:.2f}]: {e}", exc_info=True)
            continue
        if wav.size == 0:
            continue

        emb = embedder({"waveform": torch.tensor(wav)[None], "sample_rate": 16000})
        emb = _to_numpy_flat(emb)
        emb /= np.linalg.norm(emb)
        seg["__embed__"] = emb

    # 2) Build cluster centroids
    clusters = defaultdict(list)
    for seg in segments:
        emb = seg.get("__embed__")
        if isinstance(emb, np.ndarray) and emb.ndim == 1:
            clusters[seg["speaker"]].append(emb)

    centroids = {}
    for lbl, mats in clusters.items():
        if not mats:
            continue
        dims = {emb.shape[0] for emb in mats}
        if len(dims) != 1:
            logger.warning(f"Inconsistent embedding dims for '{lbl}': {dims}, skipping centroid.")
            continue
        mat_stack = np.vstack(mats)
        mean_emb = mat_stack.mean(axis=0)
        centroid = mean_emb / np.linalg.norm(mean_emb)
        centroids[lbl] = centroid

    for lbl, centroid in centroids.items():
        log_data["centroids"][lbl] = centroid.tolist()

    # 3) Relabel segments based on centroids
    for lbl, centroid in centroids.items():
        name, score = identify_speaker(centroid, known_embeddings, threshold=threshold)
        decision = {
            "original_label": lbl,
            "new_label": name,
            "similarity_score": float(score),
            "threshold": threshold,
            "relabel": name != "Unknown",
        }
        log_data["relabeling_decisions"].append(decision)

        if name == "Unknown":
            continue

        for seg in segments:
            if seg["speaker"] == lbl:
                seg["speaker"] = name
                seg["similarity"] = float(score)

    # 4) Cleanup temporary embeddings
    for seg in segments:
        seg.pop("__embed__", None)
        seg["start"] = float(seg["start"])
        seg["end"] = float(seg["end"])
        seg.setdefault("similarity", None)

    return output, log_data
