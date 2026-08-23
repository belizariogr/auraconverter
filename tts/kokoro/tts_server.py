"""
HTTP TTS server wrapping Kokoro ONNX for AuraReader.

Same API as the Qwen servers: /health, /tts, /tts/cancel, /tts/unload.
Ignores ICL fields (refAudioPath / refText / skipIcl / instruct).

GPU: prefers CUDA / MIGraphX (AMD ROCm) / DirectML / CoreML when the
installed onnxruntime build exposes them. Override with AURA_ONNX_PROVIDER
or ONNX_PROVIDER (e.g. CPUExecutionProvider).
"""

from __future__ import annotations

import argparse
import base64
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

DEFAULT_SAMPLE_RATE = 24000
PORT = int(os.environ.get("QWEN_TTS_PORT", os.environ.get("TTS_PORT", "8765")))
HOST = os.environ.get("QWEN_TTS_HOST", "127.0.0.1")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

MODEL_DIR = os.environ.get(
    "KOKORO_MODEL_DIR",
    os.path.join(SCRIPT_DIR, "models"),
)
ONNX_NAME = os.environ.get("KOKORO_ONNX_NAME", "kokoro-v1.0.onnx")
VOICES_NAME = os.environ.get("KOKORO_VOICES_NAME", "voices-v1.0.bin")
DEFAULT_VOICE = os.environ.get("KOKORO_VOICE", "af_heart")
DEFAULT_LANGUAGE = os.environ.get("KOKORO_LANGUAGE", "en-us")

KOKORO_ONNX_LANG_MAP = {
    "auto": "en-us",
    "a": "en-us",
    "en": "en-us",
    "en-us": "en-us",
    "english": "en-us",
    "b": "en-gb",
    "en-gb": "en-gb",
    "p": "pt-br",
    "pt": "pt-br",
    "pt-br": "pt-br",
    "portuguese": "pt-br",
    "e": "es",
    "es": "es",
    "es-es": "es",
    "spanish": "es",
    "f": "fr-fr",
    "fr": "fr-fr",
    "fr-fr": "fr-fr",
    "french": "fr-fr",
    "i": "it",
    "it": "it",
    "it-it": "it",
    "italian": "it",
    "j": "ja",
    "ja": "ja",
    "ja-jp": "ja",
    "japanese": "ja",
    "z": "zh",
    "zh": "zh",
    "zh-cn": "zh",
    "chinese": "zh",
    "h": "hi",
    "hi": "hi",
    "hi-in": "hi",
    "hindi": "hi",
}


def resolve_language(language: Optional[str]) -> str:
    raw = (language or DEFAULT_LANGUAGE).strip() or DEFAULT_LANGUAGE
    key = raw.lower().replace("_", "-")
    return KOKORO_ONNX_LANG_MAP.get(key, DEFAULT_LANGUAGE)


DEFAULT_SPEED = float(os.environ.get("KOKORO_SPEED", "1.0"))

# Prefer GPU EPs first; CPU always last as fallback.
_GPU_PROVIDER_PRIORITY = (
    "CUDAExecutionProvider",
    "MIGraphXExecutionProvider",
    "ROCMExecutionProvider",
    "DmlExecutionProvider",
    "CoreMLExecutionProvider",
)

# Curated English subset (matches AuraReader UI catalog).
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

_kokoro = None
_active_providers: list[str] = []
_model_lock = threading.Lock()
_cancel_jobs: set[str] = set()
_cancel_lock = threading.Lock()
_warmup_lock = threading.Lock()
_warmup_state: dict[str, Any] = {
    "running": False,
    "done": False,
    "current": 0,
    "total": 0,
    "phase": "",
    "steps": [],
    "error": None,
    "elapsedMs": 0,
}

# English snippets of increasing length → different ONNX token shapes for MIGraphX cache.
_WARMUP_TEXTS = [
    "Hi.",
    "Hello there, how are you today?",
    (
        "The quick brown fox jumps over the lazy dog near the river bank "
        "while birds sing in the morning light."
    ),
    (
        "Once upon a time in a quiet town by the sea, a young reader opened a book "
        "and heard every page spoken aloud with care, patience, and a steady voice "
        "that carried through the evening air."
    ),
    (
        "Chapter one begins with a long stretch of narrative prose intended to exercise "
        "the text to speech pipeline across a broader phoneme window. We mention rivers, "
        "mountains, markets, libraries, and distant cities so the model sees varied "
        "sounds. The goal is simply to precompile common input lengths for the GPU graph "
        "so later narration feels responsive instead of stuck on Model Compile Begin."
    ),
    (
        "In the second half of this longer warm-up paragraph we keep adding clauses so "
        "the tokenizer produces a denser sequence. Names like Michael, Sarah, Bella, and "
        "Adam appear alongside places such as London, Boston, and Kyoto. Numbers one two "
        "three four five six seven eight nine ten help fill the remaining room without "
        "needing meaning. This should cover chunk sizes typical of AuraReader narration."
    ),
]


def _ensure_compile_parallel_env() -> None:
    """Use all CPU cores for MIGraphX GPU code compilation when unset."""
    if os.environ.get("MIGRAPHX_GPU_COMPILE_PARALLEL"):
        return
    try:
        n = os.cpu_count() or 4
    except Exception:
        n = 4
    os.environ["MIGRAPHX_GPU_COMPILE_PARALLEL"] = str(max(1, n))
    print(f"[kokoro] MIGRAPHX_GPU_COMPILE_PARALLEL={os.environ['MIGRAPHX_GPU_COMPILE_PARALLEL']}")


def onnx_path() -> str:
    return os.path.join(MODEL_DIR, ONNX_NAME)


def voices_path() -> str:
    return os.path.join(MODEL_DIR, VOICES_NAME)


def assets_ready() -> bool:
    try:
        return (
            os.path.isfile(onnx_path())
            and os.path.getsize(onnx_path()) > 1_000_000
            and os.path.isfile(voices_path())
            and os.path.getsize(voices_path()) > 1_000_000
        )
    except OSError:
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
    # Accept any kokoro voice id present in the bin (fallback keep raw).
    if lowered.startswith(("af_", "am_", "bf_", "bm_", "ef_", "em_", "ff_", "jf_", "jm_", "pf_", "pm_", "zf_", "zm_")):
        return lowered
    return DEFAULT_VOICE if DEFAULT_VOICE in SPEAKERS else "af_heart"


def resolve_providers() -> list[str]:
    """Pick ONNX Runtime execution providers (GPU first when available)."""
    import onnxruntime as ort

    available = list(ort.get_available_providers())
    device_pref = (os.environ.get("KOKORO_DEVICE") or "").strip().lower()
    forced = (
        os.environ.get("AURA_ONNX_PROVIDER")
        or os.environ.get("ONNX_PROVIDER")
        or ""
    ).strip()
    if device_pref == "cpu" and not forced:
        forced = "CPUExecutionProvider"
    if forced:
        providers = [forced]
        if "CPUExecutionProvider" not in providers:
            providers.append("CPUExecutionProvider")
        return providers

    preferred: list[str] = []
    for name in _GPU_PROVIDER_PRIORITY:
        if name in available:
            preferred.append(name)
    if "CPUExecutionProvider" not in preferred:
        preferred.append("CPUExecutionProvider")
    return preferred


def provider_label(providers: list[str]) -> str:
    primary = providers[0] if providers else "CPUExecutionProvider"
    if primary.startswith("CUDA"):
        return "cuda"
    if primary.startswith("MIGraphX") or primary.startswith("ROCM"):
        return "rocm"
    if primary.startswith("Dml"):
        return "directml"
    if primary.startswith("CoreML"):
        return "coreml"
    return "cpu"


def build_provider_list(providers: list[str]) -> list:
    """ORT providers list; attach MIGraphX cache dir so compile is reused."""
    _ensure_compile_parallel_env()
    cache_dir = (
        os.environ.get("ORT_MIGRAPHX_MODEL_CACHE_PATH")
        or os.environ.get("ORT_MIGRAPHX_CACHE_PATH")
        or os.path.join(os.path.expanduser("~"), ".cache", "aura-kokoro-ort")
    )
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError:
        pass

    out: list = []
    for name in providers:
        if name == "MIGraphXExecutionProvider":
            out.append(
                (
                    name,
                    {
                        "device_id": 0,
                        # Persist .mxr compiles across runs / input-shape variants.
                        "migraphx_model_cache_dir": cache_dir,
                    },
                )
            )
            print(f"[kokoro] MIGraphX model cache: {cache_dir}")
            print(
                "[kokoro] Nota: a 1ª inferência (e cada tamanho de input novo) "
                "compila o grafo — pode demorar minutos com pouca uso de GPU. "
                "Use POST /tts/warmup (ou Aquecer GPU na UI) para pré-compilar."
            )
        else:
            out.append(name)
    return out


def ensure_model():
    global _kokoro, _active_providers
    with _model_lock:
        if _kokoro is not None:
            return _kokoro
        if not assets_ready():
            raise RuntimeError(
                f"Kokoro assets missing under {MODEL_DIR}. "
                f"Expected {ONNX_NAME} and {VOICES_NAME}."
            )
        import onnxruntime as ort
        from kokoro_onnx import Kokoro

        providers = resolve_providers()
        provider_list = build_provider_list(providers)
        print(f"[kokoro] Loading ONNX from {onnx_path()} ...")
        print(f"[kokoro] Available providers: {ort.get_available_providers()}")
        print(f"[kokoro] Using providers: {providers}")

        # kokoro-onnx defaults to CPU-only; build the session ourselves so GPU EPs work.
        # ORT may silently fall back to CPU if a GPU EP fails to load (missing libmigraphx).
        session = ort.InferenceSession(onnx_path(), providers=provider_list)
        _active_providers = list(session.get_providers())
        print(f"[kokoro] Session providers: {_active_providers}")
        wanted_gpu = [p for p in providers if p != "CPUExecutionProvider"]
        if wanted_gpu and not any(p in _active_providers for p in wanted_gpu):
            print(
                "[kokoro] WARNING: GPU provider requested but session is CPU-only. "
                "On AMD Arch: sudo pacman -S migraphx protobuf abseil-cpp "
                "(needs libprotobuf.so.35 + matching abseil), then restart AuraReader."
            )

        try:
            _kokoro = Kokoro.from_session(session, voices_path())
        except Exception as exc:
            # Older kokoro-onnx without from_session — fall back via ONNX_PROVIDER.
            print(f"[kokoro] from_session unavailable ({exc}); using Kokoro() + ONNX_PROVIDER")
            os.environ["ONNX_PROVIDER"] = providers[0]
            _kokoro = Kokoro(onnx_path(), voices_path())
            _active_providers = providers[:1]

        print("[kokoro] Model loaded")
        return _kokoro


def unload_model() -> bool:
    global _kokoro, _active_providers
    with _model_lock:
        if _kokoro is None:
            return False
        _kokoro = None
        _active_providers = []
        print("[kokoro] Model unloaded.")
        return True


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


def synthesize(
    text: str, voice: str, speed: float, language: Optional[str] = None
) -> tuple[np.ndarray, int]:
    model = ensure_model()
    lang = resolve_language(language)
    samples, sample_rate = model.create(text, voice=voice, speed=speed, lang=lang)
    return np.asarray(samples, dtype=np.float32), int(sample_rate or DEFAULT_SAMPLE_RATE)


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


def run_warmup(voice: Optional[str] = None) -> dict[str, Any]:
    """Synthesize several lengths to populate MIGraphX shape caches."""
    with _warmup_lock:
        if _warmup_state.get("running"):
            return {"started": False, "alreadyRunning": True, **warmup_status()}
        _warmup_state.update(
            {
                "running": True,
                "done": False,
                "current": 0,
                "total": len(_WARMUP_TEXTS),
                "phase": "Carregando modelo…",
                "steps": [],
                "error": None,
                "elapsedMs": 0,
            }
        )

    voice_id = resolve_voice(voice)
    started = time.time()

    def worker() -> None:
        try:
            ensure_model()
            steps: list[dict[str, Any]] = []
            for i, text in enumerate(_WARMUP_TEXTS):
                _warmup_state["current"] = i + 1
                _warmup_state["phase"] = (
                    f"Compilando tamanho {i + 1}/{len(_WARMUP_TEXTS)} "
                    f"({len(text)} chars)…"
                )
                print(f"[kokoro] warmup {i + 1}/{len(_WARMUP_TEXTS)} chars={len(text)}")
                t0 = time.time()
                synthesize(text, voice_id, DEFAULT_SPEED)
                ms = int((time.time() - t0) * 1000)
                steps.append({"index": i + 1, "chars": len(text), "ms": ms})
                _warmup_state["steps"] = list(steps)
                _warmup_state["elapsedMs"] = int((time.time() - started) * 1000)
                print(f"[kokoro] warmup step {i + 1} done in {ms}ms")
            _warmup_state["phase"] = "Concluído"
            _warmup_state["done"] = True
            _warmup_state["error"] = None
        except Exception as exc:
            _warmup_state["error"] = str(exc)
            _warmup_state["phase"] = "Falhou"
            _warmup_state["done"] = False
            print(f"[kokoro] warmup failed: {exc}")
        finally:
            _warmup_state["running"] = False
            _warmup_state["elapsedMs"] = int((time.time() - started) * 1000)

    threading.Thread(target=worker, name="kokoro-warmup", daemon=True).start()
    return {"started": True, "alreadyRunning": False, **warmup_status()}


def warmup_status() -> dict[str, Any]:
    return {
        "running": bool(_warmup_state.get("running")),
        "done": bool(_warmup_state.get("done")),
        "current": int(_warmup_state.get("current") or 0),
        "total": int(_warmup_state.get("total") or 0),
        "phase": str(_warmup_state.get("phase") or ""),
        "steps": list(_warmup_state.get("steps") or []),
        "error": _warmup_state.get("error"),
        "elapsedMs": int(_warmup_state.get("elapsedMs") or 0),
        "device": provider_label(_active_providers or resolve_providers()),
        "onnxProviders": list(_active_providers or []),
    }


class WarmupRequest(BaseModel):
    voice: Optional[str] = None


class CancelRequest(BaseModel):
    jobId: Optional[str] = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if os.environ.get("QWEN_TTS_PRELOAD", "0") in ("1", "true", "True"):
        try:
            ensure_model()
        except Exception as exc:
            print(f"[kokoro] Preload skipped: {exc}")
    print(
        f"[kokoro] Server ready (lazy model load; model_dir={MODEL_DIR}, "
        f"defaultVoice={DEFAULT_VOICE})"
    )
    yield
    unload_model()


app = FastAPI(title="AuraReader Kokoro TTS", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    import onnxruntime as ort

    providers = _active_providers or resolve_providers()
    requested = (os.environ.get("KOKORO_DEVICE") or "gpu").strip().lower()
    actual = provider_label(providers)
    fallback = requested == "gpu" and actual == "cpu" and _kokoro is not None
    return {
        "ready": True,
        "modelLoaded": _kokoro is not None,
        "provider": "kokoro",
        "backend": "onnx",
        "device": actual,
        "kokoroDevice": requested,
        "gpuFallback": fallback,
        "onnxProviders": providers,
        "availableProviders": list(ort.get_available_providers()),
        "model": ONNX_NAME,
        "icl": False,
        "sampleRate": DEFAULT_SAMPLE_RATE,
        "speakers": sorted(SPEAKERS),
        "modelDir": MODEL_DIR,
        "assetsReady": assets_ready(),
        "warmup": warmup_status(),
    }


@app.get("/tts/warmup")
def tts_warmup_get() -> dict[str, Any]:
    return warmup_status()


@app.post("/tts/warmup")
def tts_warmup_post(req: WarmupRequest = WarmupRequest()) -> dict[str, Any]:
    device_pref = (os.environ.get("KOKORO_DEVICE") or "gpu").strip().lower()
    if device_pref == "cpu":
        return {
            "started": False,
            "skipped": True,
            "message": "Warm-up de GPU ignorado (KOKORO_DEVICE=cpu).",
            **warmup_status(),
        }
    return run_warmup(req.voice)


@app.post("/tts")
def tts(req: TtsRequest) -> dict[str, Any]:
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    voice = resolve_voice(req.voice)
    language = resolve_language(req.language)
    job_id = req.jobId
    clear_cancel(job_id)

    print(
        f"[kokoro] /tts voice={voice!r} lang={language!r} "
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
        audio, sample_rate = synthesize(text, voice, DEFAULT_SPEED, language)
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
    parser = argparse.ArgumentParser(description="AuraReader Kokoro TTS server")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
