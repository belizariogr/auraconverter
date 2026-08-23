"""
HTTP TTS server wrapping Kokoro via MLX (Apple Silicon) for AuraReader.

Same API as the ONNX Kokoro server: /health, /tts, /tts/cancel, /tts/unload.
Ignores ICL fields (refAudioPath / refText / skipIcl / instruct).

Weights: mlx-community/Kokoro-82M-bf16 (full quality, not quantized).
"""

from __future__ import annotations

import argparse
import base64
import os
import threading
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

DEFAULT_SAMPLE_RATE = 24000
PORT = int(os.environ.get("QWEN_TTS_PORT", os.environ.get("TTS_PORT", "8765")))
HOST = os.environ.get("QWEN_TTS_HOST", "127.0.0.1")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.environ.get(
    "KOKORO_MODEL_DIR",
    os.path.join(SCRIPT_DIR, "models", "Kokoro-82M-bf16"),
)
DEFAULT_VOICE = os.environ.get("KOKORO_VOICE", "af_heart")
DEFAULT_SPEED = float(os.environ.get("KOKORO_SPEED", "1.0"))
# Kokoro MLX lang codes: a=American English, b=British, …
DEFAULT_LANG = os.environ.get("KOKORO_LANG_CODE", "a")

# Catalog ids from narrationLanguage.ts plus mlx-audio letters.
KOKORO_MLX_LANG_MAP = {
    "auto": "a",
    "en": "a",
    "en-us": "a",
    "english": "a",
    "en-gb": "b",
    "pt": "p",
    "pt-br": "p",
    "portuguese": "p",
    "es": "e",
    "es-es": "e",
    "spanish": "e",
    "fr": "f",
    "fr-fr": "f",
    "french": "f",
    "it": "i",
    "it-it": "i",
    "italian": "i",
    "ja": "j",
    "ja-jp": "j",
    "japanese": "j",
    "zh": "z",
    "zh-cn": "z",
    "chinese": "z",
    "hi": "h",
    "hi-in": "h",
    "hindi": "h",
}


def resolve_lang_code(language: Optional[str]) -> str:
    raw = (language or DEFAULT_LANG).strip() or DEFAULT_LANG
    if len(raw) == 1 and raw.lower() in "abefhijpz":
        return raw.lower()
    key = raw.lower().replace("_", "-")
    return KOKORO_MLX_LANG_MAP.get(key, DEFAULT_LANG)


HF_REPO = os.environ.get("KOKORO_MLX_REPO", "mlx-community/Kokoro-82M-bf16")

SPEAKERS = {
    "af_heart",
    "af_sarah",
    "af_bella",
    "af_nicole",
    "am_adam",
    "am_michael",
    "am_fenrir",
    "am_puck",
}

VOICE_ALIASES = {
    "Heart": "af_heart",
    "Sarah": "af_sarah",
    "Bella": "af_bella",
    "Nicole": "af_nicole",
    "Adam": "am_adam",
    "Michael": "am_michael",
    "Fenrir": "am_fenrir",
    "Puck": "am_puck",
    "Vivian": "af_heart",
    "Serena": "af_sarah",
    "Ryan": "am_adam",
    "Aiden": "am_michael",
}

_model = None
_model_lock = threading.Lock()
_cancel_jobs: set[str] = set()
_cancel_lock = threading.Lock()


def assets_ready() -> bool:
    """True when local MLX weights (safetensors) are present under MODEL_DIR."""
    if not os.path.isdir(MODEL_DIR):
        return False
    try:
        for root, _dirs, files in os.walk(MODEL_DIR):
            for name in files:
                if name.endswith((".safetensors", ".npz")):
                    path = os.path.join(root, name)
                    if os.path.getsize(path) > 1_000_000:
                        return True
    except OSError:
        return False
    return False


def resolve_voice(name: Optional[str]) -> str:
    if not name:
        return DEFAULT_VOICE if DEFAULT_VOICE in SPEAKERS else "af_heart"
    if name in SPEAKERS:
        return name
    alias = VOICE_ALIASES.get(name) or VOICE_ALIASES.get(name.strip())
    if alias and alias in SPEAKERS:
        return alias
    lowered = name.strip().lower().replace(" ", "_")
    if lowered in SPEAKERS:
        return lowered
    if lowered.startswith(
        ("af_", "am_", "bf_", "bm_", "ef_", "em_", "ff_", "jf_", "jm_", "pf_", "pm_", "zf_", "zm_")
    ):
        return lowered
    return DEFAULT_VOICE if DEFAULT_VOICE in SPEAKERS else "af_heart"


def apply_device_preference() -> str:
    """Honor KOKORO_DEVICE=cpu|gpu for MLX default device."""
    import mlx.core as mx

    pref = (os.environ.get("KOKORO_DEVICE") or "gpu").strip().lower()
    if pref == "cpu":
        mx.set_default_device(mx.cpu)
        return "cpu"
    # Prefer GPU / Metal when available.
    try:
        mx.set_default_device(mx.gpu)
        return "mlx"
    except Exception:
        mx.set_default_device(mx.cpu)
        return "cpu"


def ensure_model():
    global _model
    with _model_lock:
        if _model is not None:
            return _model
        if not assets_ready():
            raise RuntimeError(
                f"Kokoro MLX assets missing under {MODEL_DIR}. "
                f"Expected safetensors from {HF_REPO}."
            )
        from mlx_audio.tts.utils import load_model

        device = apply_device_preference()
        print(f"[kokoro-mlx] Loading from {MODEL_DIR} (device={device}) …")
        _model = load_model(MODEL_DIR)
        # HF repo id for voice snapshot fallback; local voices preferred in synthesize().
        try:
            _model.repo_id = HF_REPO
        except Exception:
            pass
        print("[kokoro-mlx] Model loaded")
        return _model


def unload_model() -> bool:
    global _model
    with _model_lock:
        if _model is None:
            return False
        _model = None
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:
            pass
        print("[kokoro-mlx] Model unloaded.")
        return True


def float_to_pcm16_b64(audio: np.ndarray) -> str:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not audio.size:
        return ""

    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    # Preserve the model's original level. Only attenuate true overshoots so the
    # int16 conversion cannot wrap; never hard-clip or boost the waveform.
    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        audio *= np.float32(0.999 / peak)

    pcm = np.rint(audio * 32767.0).astype(np.int16)
    return base64.b64encode(pcm.tobytes()).decode("ascii")


def job_cancelled(job_id: Optional[str]) -> bool:
    if not job_id:
        return False
    with _cancel_lock:
        return job_id in _cancel_jobs


def clear_cancel(job_id: Optional[str]) -> None:
    if not job_id:
        return
    with _cancel_lock:
        _cancel_jobs.discard(job_id)


def _to_numpy(audio: Any) -> np.ndarray:
    if hasattr(audio, "tolist"):
        return np.asarray(audio.tolist(), dtype=np.float32).reshape(-1)
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def resolve_voice_arg(voice: str) -> str:
    """Prefer local voices/*.safetensors from the downloaded model tree."""
    local = os.path.join(MODEL_DIR, "voices", f"{voice}.safetensors")
    if os.path.isfile(local):
        return local
    return voice


@contextmanager
def _hold_mlx_cache():
    """Skip mlx-audio's mx.clear_cache() after every Kokoro segment.

    generate() wipes the Metal cache between newline/phoneme windows. On M5
    that recycles Neural Accelerator buffers with leftover data — first
    window OK, the next one sounds destroyed. M2 is unaffected.
    """
    try:
        import mlx.core as mx
    except ImportError:
        yield
        return
    if not hasattr(mx, "clear_cache"):
        yield
        return
    original = mx.clear_cache
    mx.clear_cache = lambda: None  # type: ignore[method-assign]
    try:
        yield
    finally:
        mx.clear_cache = original  # type: ignore[method-assign]
        try:
            if hasattr(mx, "synchronize"):
                mx.synchronize()
            original()
        except Exception:
            pass


def synthesize(
    text: str,
    voice: str,
    speed: float,
    job_id: Optional[str] = None,
    language: Optional[str] = None,
) -> tuple[np.ndarray, int]:
    model = ensure_model()
    chunks: list[np.ndarray] = []
    sample_rate = DEFAULT_SAMPLE_RATE
    voice_arg = resolve_voice_arg(voice)
    lang_code = resolve_lang_code(language)
    # Do not split on newlines: each split is a new generate window + cache wipe.
    with _hold_mlx_cache():
        for result in model.generate(
            text,
            voice=voice_arg,
            speed=speed,
            lang_code=lang_code,
            split_pattern=None,
        ):
            if job_cancelled(job_id):
                break
            sample_rate = int(getattr(result, "sample_rate", None) or sample_rate)
            chunks.append(_to_numpy(result.audio))
    if not chunks:
        raise RuntimeError("Kokoro MLX produced empty audio.")
    return np.concatenate(chunks), sample_rate


class TtsRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice: Optional[str] = None
    language: Optional[str] = None
    instruct: Optional[str] = None
    temperature: Optional[float] = None
    refAudioPath: Optional[str] = None
    refText: Optional[str] = None
    skipIcl: Optional[bool] = None
    jobId: Optional[str] = None
    seed: Optional[int] = None


class CancelRequest(BaseModel):
    jobId: Optional[str] = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if os.environ.get("QWEN_TTS_PRELOAD", "0") in ("1", "true", "True"):
        try:
            ensure_model()
        except Exception as exc:
            print(f"[kokoro-mlx] Preload skipped: {exc}")
    print(
        f"[kokoro-mlx] Server ready (lazy model load; model_dir={MODEL_DIR}, "
        f"defaultVoice={DEFAULT_VOICE})"
    )
    yield
    unload_model()


app = FastAPI(title="AuraReader Kokoro TTS (MLX)", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    requested = (os.environ.get("KOKORO_DEVICE") or "gpu").strip().lower()
    device = "cpu" if requested == "cpu" else "mlx"
    return {
        "ready": True,
        "modelLoaded": _model is not None,
        "provider": "kokoro",
        "backend": "mlx",
        "device": device,
        "kokoroDevice": requested,
        "gpuFallback": False,
        "onnxProviders": [],
        "availableProviders": ["MLX"],
        "model": os.path.basename(MODEL_DIR.rstrip(os.sep)) or HF_REPO,
        "icl": False,
        "sampleRate": DEFAULT_SAMPLE_RATE,
        "speakers": sorted(SPEAKERS),
        "modelDir": MODEL_DIR,
        "assetsReady": assets_ready(),
        "warmup": {
            "running": False,
            "done": False,
            "current": 0,
            "total": 0,
            "phase": "",
            "steps": [],
            "error": None,
            "elapsedMs": 0,
            "device": device,
            "onnxProviders": [],
            "skipped": True,
            "message": "Warm-up MIGraphX não se aplica ao backend MLX.",
        },
    }


@app.get("/tts/warmup")
def tts_warmup_get() -> dict[str, Any]:
    return health()["warmup"]


@app.post("/tts/warmup")
def tts_warmup_post() -> dict[str, Any]:
    return {
        "started": False,
        "skipped": True,
        "message": "Warm-up de GPU (MIGraphX) não se aplica ao Kokoro MLX.",
        **health()["warmup"],
    }


@app.post("/tts")
def tts(req: TtsRequest) -> dict[str, Any]:
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    voice = resolve_voice(req.voice)
    language = resolve_lang_code(req.language)
    job_id = req.jobId
    clear_cancel(job_id)

    print(
        f"[kokoro-mlx] /tts voice={voice!r} lang={language!r} "
        f"jobId={job_id or '-'} chars={len(text)}"
    )

    if job_cancelled(job_id):
        clear_cancel(job_id)
        return {
            "sampleRate": DEFAULT_SAMPLE_RATE,
            "audioData": "",
            "format": "pcm_s16le",
            "cancelled": True,
            "voice": voice,
            "icl": False,
        }

    try:
        audio, sample_rate = synthesize(
            text, voice, DEFAULT_SPEED, job_id=job_id, language=language
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    cancelled = job_cancelled(job_id)
    clear_cancel(job_id)
    if cancelled:
        return {
            "sampleRate": sample_rate,
            "audioData": "",
            "format": "pcm_s16le",
            "cancelled": True,
            "voice": voice,
            "icl": False,
        }

    return {
        "sampleRate": sample_rate,
        "audioData": float_to_pcm16_b64(audio),
        "format": "pcm_s16le",
        "cancelled": False,
        "voice": voice,
        "icl": False,
    }


@app.post("/tts/cancel")
def tts_cancel(req: CancelRequest) -> dict[str, Any]:
    job_id = (req.jobId or "").strip()
    if job_id:
        with _cancel_lock:
            _cancel_jobs.add(job_id)
    return {"success": True, "message": "Cancel requested"}


@app.post("/tts/unload")
def tts_unload() -> dict[str, Any]:
    unloaded = unload_model()
    return {"success": True, "unloaded": unloaded}


def main() -> None:
    parser = argparse.ArgumentParser(description="AuraReader Kokoro TTS (MLX)")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
