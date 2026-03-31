# speaker_profiles.py  ---------------------------------------------
import os, tempfile, requests, numpy as np, torch, librosa
from scipy.spatial.distance import cdist

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_EMBED = None  # lazy-loaded
_CACHE = {}    # name -> 512-D vector


def _get_embed_model():
    """Lazy-load pyannote embedding model on first use."""
    global _EMBED
    if _EMBED is None:
        from pyannote.audio import Inference
        _EMBED = Inference(
            "pyannote/embedding",
            device=_DEVICE,
            use_auth_token=os.getenv("HF_TOKEN"),
        )
    return _EMBED


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x)


def load_embeddings(profiles):
    """
    >>> load_embeddings([{"name":"alice","url":"https://…/alice.wav"}, …])
    returns {'alice': 512-D np.array, …}
    """
    model = _get_embed_model()
    out = {}
    for p in profiles:
        name, url = p["name"], p["url"]
        if name in _CACHE:
            out[name] = _CACHE[name]
            continue

        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            tmp.write(requests.get(url, timeout=30).content)
            tmp.flush()
            wav, _ = librosa.load(tmp.name, sr=16_000, mono=True)
            vec = model(torch.tensor(wav).unsqueeze(0)).cpu().numpy().flatten()
            vec = _l2(vec)
            _CACHE[name] = vec
            out[name] = vec
    return out


# ---------------------------------------------------------------------
# 2)  Replace diarization labels with closest profile name
# ---------------------------------------------------------------------
def relabel(diarize_df, transcription, embeds, threshold=0.75):
    names = list(embeds.keys())
    vecstack = np.stack(list(embeds.values()))

    for seg in transcription["segments"]:
        dia_spk = seg.get("speaker")
        if not dia_spk:
            continue

        word_vecs = [w.get("embedding")
                     for w in seg.get("words", [])
                     if w.get("speaker") == dia_spk and w.get("embedding") is not None]

        if not word_vecs:
            continue

        centroid = np.mean(word_vecs, axis=0, keepdims=True)
        sim = 1 - cdist(centroid, vecstack, metric="cosine")
        best_idx = int(sim.argmax())
        if sim[0, best_idx] >= threshold:
            real = names[best_idx]
            seg["speaker"] = real
            seg["similarity"] = float(sim[0, best_idx])
            for w in seg.get("words", []):
                w["speaker"] = real
    return transcription
