#from cog import BasePredictor, Input, Path, BaseModel
try:
    # Prefer real cog if present (e.g. when running locally)
    from cog import BasePredictor, Input, Path, BaseModel
except ImportError:                          # pragma: no cover
    from cog_stub import BasePredictor, Input, Path, BaseModel

# cuDNN 8 is installed in this image (alongside cuDNN 9 from torch).
# onnxruntime should find libcudnn_ops_infer.so.8 and use CUDA for VAD.
# No monkey-patching needed — let onnxruntime use its default providers.

# Patch huggingface_hub functions ONLY: newer hub needs 'token' not 'use_auth_token'
# Do NOT patch pyannote Pipeline — it still uses 'use_auth_token' internally
import huggingface_hub as _hfh
for _fn_name in ('hf_hub_download', 'snapshot_download', 'model_info'):
    _orig_fn = getattr(_hfh, _fn_name, None)
    if _orig_fn:
        def _make_patched(orig):
            def _patched(*args, **kwargs):
                if 'use_auth_token' in kwargs:
                    kwargs['token'] = kwargs.pop('use_auth_token')
                return orig(*args, **kwargs)
            return _patched
        setattr(_hfh, _fn_name, _make_patched(_orig_fn))

from pydub import AudioSegment
from typing import Any
from whisperx.audio import N_SAMPLES, log_mel_spectrogram
from scipy.spatial.distance import cosine
import gc
import math
import os
import shutil
import whisperx
import tempfile
import time
import torch
import speaker_processing


import logging
import sys
logger = logging.getLogger("predict")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_formatter = logging.Formatter(
    "%(asctime)s %(levelname)s [%(name)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(console_formatter)

file_handler = logging.FileHandler("container_log.txt", mode="a")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(console_formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)






torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
compute_type = "float16"  # change to "int8" if low on GPU mem (may reduce accuracy)
device = "cuda"
whisper_arch = "./models/faster-whisper-large-v3"


class Output(BaseModel):
    segments: Any
    detected_language: str


class Predictor(BasePredictor):
    def setup(self):
        source_folder = './models/vad'
        destination_folder = '../root/.cache/torch'
        file_name = 'whisperx-vad-segmentation.bin'

        os.makedirs(destination_folder, exist_ok=True)

        source_file_path = os.path.join(source_folder, file_name)
        if os.path.exists(source_file_path):
            destination_file_path = os.path.join(destination_folder, file_name)

            if not os.path.exists(destination_file_path):
                shutil.copy(source_file_path, destination_folder)

    def predict(
            self,
            audio_file: Path = Input(description="Audio file"),
            language: str = Input(
                description="ISO code of the language spoken in the audio, specify None to perform language detection",
                default=None),
            language_detection_min_prob: float = Input(
                description="If language is not specified, then the language will be detected recursively on different "
                            "parts of the file until it reaches the given probability",
                default=0
            ),
            language_detection_max_tries: int = Input(
                description="If language is not specified, then the language will be detected following the logic of "
                            "language_detection_min_prob parameter, but will stop after the given max retries. If max "
                            "retries is reached, the most probable language is kept.",
                default=5
            ),
            initial_prompt: str = Input(
                description="Optional text to provide as a prompt for the first window",
                default=None),
            batch_size: int = Input(
                description="Parallelization of input audio transcription",
                default=64),
            temperature: float = Input(
                description="Temperature to use for sampling",
                default=0),
            vad_onset: float = Input(
                description="VAD onset",
                default=0.500),
            vad_offset: float = Input(
                description="VAD offset",
                default=0.363),
            align_output: bool = Input(
                description="Aligns whisper output to get accurate word-level timestamps",
                default=False),
            diarization: bool = Input(
                description="Assign speaker ID labels",
                default=False),
            huggingface_access_token: str = Input(
                description="To enable diarization, please enter your HuggingFace token (read). You need to accept "
                            "the user agreement for the models specified in the README.",
                default=None),
            min_speakers: int = Input(
                description="Minimum number of speakers if diarization is activated (leave blank if unknown)",
                default=None),
            max_speakers: int = Input(
                description="Maximum number of speakers if diarization is activated (leave blank if unknown)",
                default=None),
            debug: bool = Input(
                description="Print out compute/inference times and memory usage information",
                default=False),
            speaker_verification: bool = Input(
                description="Enable speaker verification",
                default=False),
            speaker_samples: list = Input(
                description="List of speaker samples for verification. Each sample should be a dict with 'url' and "
                            "optional 'name' and 'file_path'. If 'name' is not provided, the file name (without "
                            "extension) is used. If 'file_path' is provided, it will be used directly.",
                default=[]
            ),
            custom_align_model: str = Input(
                description="Custom alignment model name from Hugging Face or torchaudio. If not specified, the "
                            "default model for the detected language will be used. Example: "
                            "'jonatasgrosman/wav2vec2-large-xlsr-53-german' or 'WAV2VEC2_ASR_BASE_960H'",
                default=None
            )
    ) -> Output:
        with torch.inference_mode():
            asr_options = {
                "temperatures": [temperature, 0.2, 0.4, 0.6, 0.8, 1.0] if temperature == 0 else [temperature],
                "initial_prompt": initial_prompt,
                "condition_on_previous_text": False,       # Prevents snowball hallucinations
                "no_speech_threshold": 0.6,                # Detect silence
                "compression_ratio_threshold": 2.4,        # Filter hallucinated segments (high compression)
                "beam_size": 5,                            # Better accuracy (searches more possibilities)
                # NOTE: no_repeat_ngram_size and repetition_penalty intentionally REMOVED
                # They block legitimate non-English phrases (e.g. "acha acha" in Hindi)
                # and cause missing words. Hallucinations handled by post-filter instead.
            }

            vad_options = {
                "vad_onset": vad_onset,
                "vad_offset": vad_offset
            }

            # Language handling: when language=None, let Whisper auto-detect per chunk.
            # Do NOT run recursive language detection — it forces ONE language on the
            # entire file, which breaks code-switched audio (e.g. Hindi + English).
            # Whisper natively detects language per ~30s chunk when language=None.

            start_time = time.time_ns() / 1e6

            model = whisperx.load_model(whisper_arch, device, compute_type=compute_type, language=language,
                                        asr_options=asr_options, vad_options=vad_options)

            if debug:
                elapsed_time = time.time_ns() / 1e6 - start_time
                print(f"Duration to load model: {elapsed_time:.2f} ms")

            start_time = time.time_ns() / 1e6

            audio = whisperx.load_audio(audio_file)

            if debug:
                elapsed_time = time.time_ns() / 1e6 - start_time
                print(f"Duration to load audio: {elapsed_time:.2f} ms")

            start_time = time.time_ns() / 1e6

            result = model.transcribe(audio, batch_size=batch_size)
            detected_language = result["language"]

            if debug:
                elapsed_time = time.time_ns() / 1e6 - start_time
                print(f"Duration to transcribe: {elapsed_time:.2f} ms")

            gc.collect()
            torch.cuda.empty_cache()
            del model

            # Always align when diarization is enabled — word-level timestamps
            # are critical for accurate speaker assignment
            needs_align = align_output or diarization
            if needs_align:
                if (detected_language in whisperx.alignment.DEFAULT_ALIGN_MODELS_TORCH or
                    detected_language in whisperx.alignment.DEFAULT_ALIGN_MODELS_HF or
                    custom_align_model is not None):
                    result = align(audio, result, debug, custom_align_model)
                else:
                    print(f"Warning: Cannot align for language {detected_language}. "
                          f"Diarization accuracy may be reduced without word-level timestamps.")

            if diarization:
                result = diarize(audio_file, result, debug, huggingface_access_token, min_speakers, max_speakers)

            if debug:
                print(f"max gpu memory allocated over runtime: {torch.cuda.max_memory_reserved() / (1024 ** 3):.2f} GB")

        return Output(
            segments=result["segments"],
            detected_language=detected_language
        )


def get_audio_duration(file_path):
    
    return len(AudioSegment.from_file(file_path))


def detect_language(full_audio_file_path, segments_starts, language_detection_min_prob,
                    language_detection_max_tries, asr_options, vad_options, iteration=1):
    model = whisperx.load_model(whisper_arch, device, compute_type=compute_type, asr_options=asr_options,
                                vad_options=vad_options)

    start_ms = segments_starts[iteration - 1]

    audio_segment_file_path = extract_audio_segment(full_audio_file_path, start_ms, 30000)

    audio = whisperx.load_audio(audio_segment_file_path)

    model_n_mels = model.model.feat_kwargs.get("feature_size")
    segment = log_mel_spectrogram(audio[: N_SAMPLES],
                                  n_mels=model_n_mels if model_n_mels is not None else 80,
                                  padding=0 if audio.shape[0] >= N_SAMPLES else N_SAMPLES - audio.shape[0])
    encoder_output = model.model.encode(segment)
    results = model.model.model.detect_language(encoder_output)
    language_token, language_probability = results[0][0]
    language = language_token[2:-2]

    print(f"Iteration {iteration} - Detected language: {language} ({language_probability:.2f})")

    audio_segment_file_path.unlink()

    gc.collect()
    torch.cuda.empty_cache()
    del model

    detected_language = {
        "language": language,
        "probability": language_probability,
        "iterations": iteration
    }

    if language_probability >= language_detection_min_prob or iteration >= language_detection_max_tries:
        return detected_language

    next_iteration_detected_language = detect_language(full_audio_file_path, segments_starts,
                                                       language_detection_min_prob, language_detection_max_tries,
                                                       asr_options, vad_options, iteration + 1)

    if next_iteration_detected_language["probability"] > detected_language["probability"]:
        return next_iteration_detected_language

    return detected_language


def extract_audio_segment(input_file_path, start_time_ms, duration_ms):
    input_file_path = Path(input_file_path) if not isinstance(input_file_path, Path) else input_file_path

    audio = AudioSegment.from_file(input_file_path)

    end_time_ms = start_time_ms + duration_ms
    extracted_segment = audio[start_time_ms:end_time_ms]

    file_extension = input_file_path.suffix

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file_path = Path(temp_file.name)
        extracted_segment.export(temp_file_path, format=file_extension.lstrip('.'))

    return temp_file_path


def distribute_segments_equally(total_duration, segments_duration, iterations):
    available_duration = total_duration - segments_duration

    if iterations > 1:
        spacing = available_duration // (iterations - 1)
    else:
        spacing = 0

    start_times = [i * spacing for i in range(iterations)]

    if iterations > 1:
        start_times[-1] = total_duration - segments_duration

    return start_times


def align(audio, result, debug, custom_align_model=None):
    start_time = time.time_ns() / 1e6

    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device, model_name=custom_align_model)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device,
                            return_char_alignments=False)

    if debug:
        elapsed_time = time.time_ns() / 1e6 - start_time
        print(f"Duration to align output: {elapsed_time:.2f} ms")

    gc.collect()
    torch.cuda.empty_cache()
    del model_a

    return result


def diarize(audio, result, debug, huggingface_access_token, min_speakers, max_speakers):
    start_time = time.time_ns() / 1e6

    # Call pyannote DIRECTLY (bypass whisperx wrapper)
    # Don't pass token — models are pre-downloaded, HF_TOKEN env var is set
    from pyannote.audio import Pipeline as PyannotePipeline
    pipeline = PyannotePipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    pipeline = pipeline.to(torch.device(device))

    # Run diarization
    import whisperx.audio
    diarize_kwargs = {}
    if min_speakers is not None:
        diarize_kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        diarize_kwargs["max_speakers"] = max_speakers
    diarize_segments = pipeline(audio, **diarize_kwargs)

    # Convert pyannote output to whisperx format for assign_word_speakers
    diarize_df = []
    for turn, _, speaker in diarize_segments.itertracks(yield_label=True):
        diarize_df.append({"start": turn.start, "end": turn.end, "speaker": speaker})
    import pandas as pd
    diarize_segments_df = pd.DataFrame(diarize_df)

    result = whisperx.assign_word_speakers(diarize_segments_df, result)

    if debug:
        elapsed_time = time.time_ns() / 1e6 - start_time
        print(f"Duration to diarize segments: {elapsed_time:.2f} ms")

    gc.collect()
    torch.cuda.empty_cache()
    del pipeline

    return result

def identify_speaker_for_segment(segment_embedding, known_embeddings, threshold=0.1):
    """
    Compare segment_embedding to known speaker embeddings using cosine similarity.
    Returns the speaker name with the highest similarity above the threshold,
    or "Unknown" if none match.
    """
    best_match = "Unknown"
    best_similarity = -1
    for speaker, known_emb in known_embeddings.items():
        similarity = 1 - cosine(segment_embedding, known_emb)
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = speaker
    if best_similarity >= threshold:
        return best_match, best_similarity
    else:
        return "Unknown", best_similarity