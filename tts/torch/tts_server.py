"""
HTTP TTS server wrapping Qwen3-TTS (PyTorch / qwen-tts) for AuraReader on Windows/Linux.

Same API as the MLX server: /health, /tts, /tts/cancel, /tts/unload.
Prefers Base (ICL voice clone) when a preview anchor exists; CustomVoice for
skipIcl / speaker-id generation.
"""

from __future__ import annotations

import argparse
import base64
import gc
import os
import threading
from contextlib import asynccontextmanager
from typing import Any, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from qwen_tts import Qwen3TTSModel

DEFAULT_SAMPLE_RATE = 24000
PORT = int(os.environ.get("QWEN_TTS_PORT", os.environ.get("DIA_PORT", "8765")))
HOST = os.environ.get("QWEN_TTS_HOST", "127.0.0.1")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BASE_MODEL_FOLDER = "Qwen3-TTS-12Hz-0.6B-Base"
CUSTOM_MODEL_FOLDER = "Qwen3-TTS-12Hz-0.6B-CustomVoice"
MODELS_DIR = os.environ.get(
    "QWEN_TTS_MODELS_DIR",
    os.path.join(SCRIPT_DIR, "models"),
)

DEFAULT_VOICE = os.environ.get("QWEN_TTS_VOICE", "vivian")
DEFAULT_LANGUAGE = os.environ.get("QWEN_TTS_LANGUAGE", "Auto")
DEFAULT_TEMPERATURE = float(os.environ.get("QWEN_TTS_TEMPERATURE", "0.3"))
DEFAULT_INSTRUCT = os.environ.get(
    "QWEN_TTS_INSTRUCT",
    (
        "Speak as a consistent, calm, neutral book narrator. "
        "Keep the same pitch, energy, emotion, and pace for every sentence. "
        "Do not sound excited, dramatic, whispery, or casual."
    ),
)
DEFAULT_REF_TEXT = os.environ.get(
    "QWEN_TTS_PREVIEW_TEXT",
    "Hello. This is a preview of my voice, reading in a calm and clear tone.",
)
VOICE_PREVIEW_DIR = os.environ.get(
    "VOICE_PREVIEW_DIR",
    os.path.join(REPO_ROOT, "assets", "voice-previews"),
)
VOICE_PREVIEW_CACHE_VERSION = os.environ.get("QWEN_TTS_PREVIEW_CACHE_VERSION", "en-v2")

SPEAKERS = {
    "vivian",
    "serena",
    "ryan",
    "aiden",
    "uncle_fu",
    "ono_anna",
    "sohee",
    "eric",
    "dylan",
}

# qwen-tts CustomVoice expects Title_Case / known display names.
SPEAKER_API_NAMES = {
    "vivian": "Vivian",
    "serena": "Serena",
    "ryan": "Ryan",
    "aiden": "Aiden",
    "uncle_fu": "Uncle_Fu",
    "ono_anna": "Ono_Anna",
    "sohee": "Sohee",
    "eric": "Eric",
    "dylan": "Dylan",
}

VOICE_ALIASES = {
    "Vivian": "vivian",
    "Serena": "serena",
    "Ryan": "ryan",
    "Aiden": "aiden",
    "Uncle_Fu": "uncle_fu",
    "Uncle Fu": "uncle_fu",
    "Ono_Anna": "ono_anna",
    "Ono Anna": "ono_anna",
    "Sohee": "sohee",
    "Eric": "eric",
    "Dylan": "dylan",
    "Ethan": "eric",
    "Chelsie": "sohee",
    "Kore": "vivian",
    "Zephyr": "ryan",
    "Puck": "aiden",
    "Charon": "eric",
    "Fenrir": "dylan",
}

LANGUAGE_MAP = {
    "auto": "Auto",
    "en": "English",
    "english": "English",
    "zh": "Chinese",
    "chinese": "Chinese",
    "ja": "Japanese",
    "japanese": "Japanese",
    "ko": "Korean",
    "korean": "Korean",
    "de": "German",
    "german": "German",
    "fr": "French",
    "french": "French",
    "ru": "Russian",
    "russian": "Russian",
    "pt": "Portuguese",
    "portuguese": "Portuguese",
    "es": "Spanish",
    "spanish": "Spanish",
    "it": "Italian",
    "italian": "Italian",
}


def resolve_voice(name: Optional[str]) -> str:
    if not name:
        return DEFAULT_VOICE if DEFAULT_VOICE in SPEAKERS else "vivian"

    if name in SPEAKERS:
        return name

    alias = VOICE_ALIASES.get(name)
    if alias and alias in SPEAKERS:
        return alias

    lowered = name.strip().lower().replace(" ", "_")
    if lowered in SPEAKERS:
        return lowered

    alias_ci = VOICE_ALIASES.get(name.strip()) or VOICE_ALIASES.get(name.strip().title())
    if alias_ci and alias_ci in SPEAKERS:
        return alias_ci

    return DEFAULT_VOICE if DEFAULT_VOICE in SPEAKERS else "vivian"


def speaker_api_name(voice: str) -> str:
    key = resolve_voice(voice)
    return SPEAKER_API_NAMES.get(key, key.replace("_", " ").title().replace(" ", "_"))


def resolve_language(language: Optional[str]) -> str:
    raw = (language or DEFAULT_LANGUAGE).strip() or DEFAULT_LANGUAGE
    mapped = LANGUAGE_MAP.get(raw.lower())
    if mapped:
        return mapped
    # Already a display name (English, Auto, …)
    if raw[:1].isupper():
        return raw
    return raw.title()


def resolve_device() -> str:
    override = (os.environ.get("AURA_TTS_DEVICE") or "").strip().lower()
    if override in ("cpu",):
        return "cpu"
    if override in ("cuda", "hip", "rocm", "gpu"):
        if torch.cuda.is_available():
            return "cuda"
        print(f"[qwen-tts] AURA_TTS_DEVICE={override} requested but CUDA/HIP unavailable; using cpu")
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def detect_accel(device: str) -> str:
    if device == "cpu" or not torch.cuda.is_available():
        return "cpu"
    hip = getattr(torch.version, "hip", None)
    if hip:
        return "rocm"
    return "cuda"


def resolve_dtype(device: str):
    if device == "cpu":
        return torch.float32
    # bf16 is widely supported on modern NVIDIA + ROCm RDNA3/4
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


DEVICE = resolve_device()
ACCEL = detect_accel(DEVICE)
DTYPE = resolve_dtype(DEVICE)

base_model: Any = None
custom_model: Any = None
active_kind: Optional[str] = None  # "base" | "custom"
model_sample_rate = DEFAULT_SAMPLE_RATE
model_tts_type = "unknown"
model_icl_capable = False
cancel_flags: dict[str, threading.Event] = {}
cancel_lock = threading.Lock()
model_lock = threading.RLock()
server_ready = False


def model_folder_present(folder_name: str) -> bool:
    return os.path.isdir(os.path.join(MODELS_DIR, folder_name))


def resolve_default_model_folder() -> str:
    env = os.environ.get("QWEN_TTS_MODEL")
    if env:
        return env
    if model_folder_present(BASE_MODEL_FOLDER):
        return BASE_MODEL_FOLDER
    return CUSTOM_MODEL_FOLDER


DEFAULT_MODEL_FOLDER = resolve_default_model_folder()


def resolve_model_path(folder_name: str) -> str:
    full_path = os.path.join(MODELS_DIR, folder_name)
    if not os.path.isdir(full_path):
        raise FileNotFoundError(
            f"Model not found at {full_path}. "
            f"Download Base for ICL: "
            f"huggingface_hub.snapshot_download('Qwen/{BASE_MODEL_FOLDER}', "
            f"local_dir='models/{BASE_MODEL_FOLDER}'). "
            f"Or keep CustomVoice at models/{CUSTOM_MODEL_FOLDER}."
        )

    snapshots_dir = os.path.join(full_path, "snapshots")
    if os.path.isdir(snapshots_dir):
        subfolders = [f for f in os.listdir(snapshots_dir) if not f.startswith(".")]
        if subfolders:
            return os.path.join(snapshots_dir, subfolders[0])
    return full_path


def float_to_pcm16_b64(audio: np.ndarray) -> str:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not audio.size:
        return ""
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    # Preserve the original level and attenuate only exceptional overshoots.
    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        audio *= np.float32(0.999 / peak)
    pcm = np.rint(audio * 32767.0).astype(np.int16)
    return base64.b64encode(pcm.tobytes()).decode("ascii")


def is_cancelled(job_id: Optional[str]) -> bool:
    if not job_id:
        return False
    with cancel_lock:
        flag = cancel_flags.get(job_id)
    return bool(flag and flag.is_set())


def preview_paths_for_voice(voice: str) -> tuple[str, str]:
    safe = resolve_voice(voice).replace(" ", "_").lower()
    wav = os.path.join(
        VOICE_PREVIEW_DIR, f"{safe}_{VOICE_PREVIEW_CACHE_VERSION}.wav"
    )
    txt = os.path.join(
        VOICE_PREVIEW_DIR, f"{safe}_{VOICE_PREVIEW_CACHE_VERSION}.txt"
    )
    return wav, txt


def load_preview_anchor(voice: str) -> tuple[Optional[str], Optional[str]]:
    wav_path, txt_path = preview_paths_for_voice(voice)
    if not (os.path.isfile(wav_path) and os.path.isfile(txt_path)):
        return None, None
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            ref_text = f.read().strip()
    except OSError:
        return None, None
    if not ref_text:
        ref_text = DEFAULT_REF_TEXT
    return wav_path, ref_text


def gpu_name() -> Optional[str]:
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return None


def _drop_model(kind: str) -> None:
    global base_model, custom_model, active_kind, model_tts_type, model_icl_capable
    if kind == "base":
        base_model = None
    elif kind == "custom":
        custom_model = None
    if active_kind == kind:
        active_kind = None
        model_tts_type = "unknown"
        model_icl_capable = False
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception as exc:
            print(f"[qwen-tts] torch.cuda.empty_cache failed: {exc}")


def _load_qwen_model(folder_name: str) -> Any:
    model_path = resolve_model_path(folder_name)
    device_map = DEVICE if DEVICE == "cpu" else "cuda:0"
    print(
        f"[qwen-tts] Loading {folder_name} from {model_path} "
        f"(device={device_map}, dtype={DTYPE}, accel={ACCEL}) ..."
    )
    loaded = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map=device_map,
        dtype=DTYPE,
        attn_implementation="sdpa",
    )
    print(f"[qwen-tts] Loaded {folder_name}")
    return loaded


def ensure_model_loaded(kind: str) -> None:
    """Load Base or CustomVoice. Caller must hold model_lock."""
    global base_model, custom_model, active_kind, model_sample_rate
    global model_tts_type, model_icl_capable, DEFAULT_MODEL_FOLDER

    if kind == "base":
        if base_model is not None:
            active_kind = "base"
            model_tts_type = "base"
            model_icl_capable = True
            DEFAULT_MODEL_FOLDER = BASE_MODEL_FOLDER
            return
        # Free the other model to limit VRAM use.
        if custom_model is not None:
            print("[qwen-tts] Unloading CustomVoice to free VRAM for Base")
            _drop_model("custom")
        base_model = _load_qwen_model(BASE_MODEL_FOLDER)
        active_kind = "base"
        model_tts_type = "base"
        model_icl_capable = True
        DEFAULT_MODEL_FOLDER = BASE_MODEL_FOLDER
        return

    if kind == "custom":
        if custom_model is not None:
            active_kind = "custom"
            model_tts_type = "custom_voice"
            model_icl_capable = False
            DEFAULT_MODEL_FOLDER = CUSTOM_MODEL_FOLDER
            return
        if base_model is not None:
            print("[qwen-tts] Unloading Base to free VRAM for CustomVoice")
            _drop_model("base")
        custom_model = _load_qwen_model(CUSTOM_MODEL_FOLDER)
        active_kind = "custom"
        model_tts_type = "custom_voice"
        model_icl_capable = False
        DEFAULT_MODEL_FOLDER = CUSTOM_MODEL_FOLDER
        return

    raise ValueError(f"Unknown model kind: {kind}")


def unload_model() -> bool:
    with model_lock:
        had = base_model is not None or custom_model is not None
        if not had:
            return False
        print("[qwen-tts] Unloading model(s)...")
        _drop_model("base")
        _drop_model("custom")
        print("[qwen-tts] Model unloaded.")
        return True


def synthesize(
    text: str,
    voice: str,
    language: str,
    instruct: str,
    temperature: float,
    ref_audio_path: Optional[str] = None,
    ref_text: Optional[str] = None,
    skip_icl: bool = False,
) -> tuple[np.ndarray, int, bool]:
    """Returns (audio, sample_rate, used_icl)."""
    # Keep newlines as pause cues (Node sends `\n\n\n` instead of `<break>` for Qwen).
    text = "\n".join(" ".join(line.split()) for line in text.split("\n")).strip(" \t")
    language = resolve_language(language)

    if skip_icl:
        ref_audio_path = None
        ref_text = None
    else:
        if not ref_audio_path or not ref_text:
            auto_wav, auto_txt = load_preview_anchor(voice)
            if not ref_audio_path:
                ref_audio_path = auto_wav
            if not ref_text:
                ref_text = auto_txt

    has_ref = bool(
        ref_audio_path and ref_text and os.path.isfile(ref_audio_path)
    )
    use_icl = bool(has_ref and not skip_icl and model_folder_present(BASE_MODEL_FOLDER))

    if use_icl:
        ensure_model_loaded("base")
        assert base_model is not None
        print(
            f"[qwen-tts] synthesize voice={voice} icl=True ref={ref_audio_path}"
        )
        wavs, sr = base_model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=ref_audio_path,
            ref_text=ref_text,
            temperature=temperature,
        )
    else:
        if not skip_icl and not has_ref and model_folder_present(BASE_MODEL_FOLDER):
            # Narration with Base prefers ICL anchors — mirror MLX behavior.
            if not model_folder_present(CUSTOM_MODEL_FOLDER):
                raise ValueError(
                    f"Missing voice preview anchor for '{voice}'. "
                    f"Generate the preview first (expected under {VOICE_PREVIEW_DIR})."
                )
        if not model_folder_present(CUSTOM_MODEL_FOLDER):
            raise ValueError(
                f"CustomVoice model not found at {os.path.join(MODELS_DIR, CUSTOM_MODEL_FOLDER)}"
            )
        ensure_model_loaded("custom")
        assert custom_model is not None
        speaker = speaker_api_name(voice)
        print(
            f"[qwen-tts] synthesize voice={voice} speaker={speaker} icl=False"
        )
        wavs, sr = custom_model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
            temperature=temperature,
        )
        use_icl = False

    sample_rate = int(sr or DEFAULT_SAMPLE_RATE)
    if not wavs:
        return np.zeros(0, dtype=np.float32), sample_rate, use_icl

    first = wavs[0]
    audio = np.asarray(first, dtype=np.float32).reshape(-1)
    return audio, sample_rate, use_icl


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global server_ready, DEFAULT_MODEL_FOLDER, DEVICE, ACCEL, DTYPE
    DEVICE = resolve_device()
    ACCEL = detect_accel(DEVICE)
    DTYPE = resolve_dtype(DEVICE)
    DEFAULT_MODEL_FOLDER = resolve_default_model_folder()
    preload = _env_flag("QWEN_TTS_PRELOAD", default=False)
    if preload:
        try:
            with model_lock:
                kind = "base" if model_folder_present(BASE_MODEL_FOLDER) else "custom"
                ensure_model_loaded(kind)
            print(
                f"[qwen-tts] Server ready (preloaded model={DEFAULT_MODEL_FOLDER}, "
                f"type={model_tts_type}, icl={model_icl_capable}, "
                f"device={DEVICE}, accel={ACCEL}, gpu={gpu_name() or '-'})"
            )
        except Exception as exc:
            print(f"[qwen-tts] WARNING: model preload failed: {exc}")
            print("[qwen-tts] Server ready, but conversion will fail until the model loads.")
    else:
        print(
            f"[qwen-tts] Server ready (lazy model load; model={DEFAULT_MODEL_FOLDER}, "
            f"device={DEVICE}, accel={ACCEL}, gpu={gpu_name() or '-'}, "
            f"defaultVoice={DEFAULT_VOICE}, previews={VOICE_PREVIEW_DIR})"
        )
    server_ready = True
    yield
    server_ready = False
    unload_model()


app = FastAPI(title="Qwen3 TTS (Torch)", lifespan=lifespan)


class TtsRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice: Optional[str] = None
    seed: Optional[int] = None
    language: Optional[str] = None
    instruct: Optional[str] = None
    temperature: Optional[float] = None
    refAudioPath: Optional[str] = None
    refText: Optional[str] = None
    skipIcl: Optional[bool] = False
    jobId: Optional[str] = None


class CancelRequest(BaseModel):
    jobId: str


@app.get("/health")
def health():
    return {
        "ready": server_ready,
        "modelLoaded": base_model is not None or custom_model is not None,
        "provider": "qwen3-tts",
        "backend": "torch",
        "model": DEFAULT_MODEL_FOLDER,
        "ttsModelType": model_tts_type,
        "icl": model_icl_capable or model_folder_present(BASE_MODEL_FOLDER),
        "sampleRate": model_sample_rate,
        "defaultVoice": DEFAULT_VOICE,
        "defaultTemperature": DEFAULT_TEMPERATURE,
        "previewDir": VOICE_PREVIEW_DIR,
        "speakers": sorted(SPEAKERS),
        "device": DEVICE,
        "accel": ACCEL,
        "gpu": gpu_name(),
    }


@app.post("/tts")
def tts(req: TtsRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    voice = resolve_voice(req.voice)
    language = (req.language or DEFAULT_LANGUAGE).strip() or DEFAULT_LANGUAGE
    instruct = (req.instruct or DEFAULT_INSTRUCT).strip() or DEFAULT_INSTRUCT
    temperature = (
        DEFAULT_TEMPERATURE if req.temperature is None else float(req.temperature)
    )
    if temperature < 0:
        temperature = 0.0
    if temperature > 1.5:
        temperature = 1.5

    ref_audio_path = (req.refAudioPath or "").strip() or None
    ref_text = (req.refText or "").strip() or None
    skip_icl = bool(req.skipIcl)
    job_id = req.jobId

    print(
        f"[qwen-tts] /tts request voice={voice!r} (from {req.voice!r}) "
        f"skipIcl={skip_icl} refAudioPath={ref_audio_path or '-'} "
        f"jobId={job_id or '-'}"
    )

    if job_id:
        with cancel_lock:
            cancel_flags[job_id] = threading.Event()

    if is_cancelled(job_id):
        if job_id:
            with cancel_lock:
                cancel_flags.pop(job_id, None)
        return {
            "sampleRate": model_sample_rate,
            "audioData": "",
            "format": "pcm_s16le",
            "cancelled": True,
            "voice": voice,
            "icl": False,
        }

    used_icl = False
    sample_rate = model_sample_rate
    audio = np.zeros(0, dtype=np.float32)

    with model_lock:
        if is_cancelled(job_id):
            if job_id:
                with cancel_lock:
                    cancel_flags.pop(job_id, None)
            return {
                "sampleRate": model_sample_rate,
                "audioData": "",
                "format": "pcm_s16le",
                "cancelled": True,
                "voice": voice,
                "icl": False,
            }

        try:
            audio, sample_rate, used_icl = synthesize(
                text,
                voice,
                language,
                instruct,
                temperature,
                ref_audio_path=ref_audio_path,
                ref_text=ref_text,
                skip_icl=skip_icl,
            )
            cancelled = is_cancelled(job_id)
        except ValueError as exc:
            if job_id:
                with cancel_lock:
                    cancel_flags.pop(job_id, None)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            if job_id:
                with cancel_lock:
                    cancel_flags.pop(job_id, None)
            raise HTTPException(
                status_code=500, detail=f"TTS generation failed: {exc}"
            ) from exc
        finally:
            if job_id:
                with cancel_lock:
                    cancel_flags.pop(job_id, None)

        if cancelled:
            return {
                "sampleRate": sample_rate,
                "audioData": "",
                "format": "pcm_s16le",
                "cancelled": True,
                "voice": voice,
                "icl": used_icl,
            }

        audio_b64 = float_to_pcm16_b64(audio) if audio.size else ""
        return {
            "sampleRate": sample_rate,
            "audioData": audio_b64,
            "format": "pcm_s16le",
            "cancelled": False,
            "voice": voice,
            "icl": used_icl,
        }


@app.post("/tts/cancel")
def cancel(req: CancelRequest):
    with cancel_lock:
        flag = cancel_flags.get(req.jobId)
        if flag is None:
            return {"success": False, "message": "No active job with that id"}
        flag.set()
    return {"success": True, "message": "Cancel requested"}


@app.post("/tts/unload")
def unload():
    unloaded = unload_model()
    return {"success": True, "unloaded": unloaded}


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS (PyTorch) HTTP server for AuraReader"
    )
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
