"""
HTTP TTS server wrapping Qwen3-TTS (MLX) for AuraReader.

Prefers the Lite Base model (0.6B) so narration can use ICL voice cloning with a
fixed preview WAV+TXT anchor (locks speaker identity across chunks).
Falls back to CustomVoice if Base is not installed.
"""

from __future__ import annotations

import argparse
import base64
import gc
import os
import threading
from contextlib import asynccontextmanager, contextmanager
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mlx_audio.tts.utils import load_model

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover
    mx = None

DEFAULT_SAMPLE_RATE = 24000
PORT = int(os.environ.get("QWEN_TTS_PORT", os.environ.get("DIA_PORT", "8765")))
HOST = os.environ.get("QWEN_TTS_HOST", "127.0.0.1")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
BASE_MODEL_FOLDER = "Qwen3-TTS-12Hz-0.6B-Base-8bit"
CUSTOM_MODEL_FOLDER = "Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit"
MODELS_DIR = os.environ.get(
    "QWEN_TTS_MODELS_DIR",
    os.path.join(SCRIPT_DIR, "models"),
)


def resolve_default_model_folder() -> str:
    env = os.environ.get("QWEN_TTS_MODEL")
    if env:
        return env
    if os.path.isdir(os.path.join(MODELS_DIR, BASE_MODEL_FOLDER)):
        return BASE_MODEL_FOLDER
    return CUSTOM_MODEL_FOLDER


DEFAULT_MODEL_FOLDER = resolve_default_model_folder()
DEFAULT_VOICE = os.environ.get("QWEN_TTS_VOICE", "vivian")
DEFAULT_LANGUAGE = os.environ.get("QWEN_TTS_LANGUAGE", "Auto")
# Low temperature keeps prosody stable; ICL anchors lock speaker identity.
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

# Speakers shared by CustomVoice / common AuraReader ids.
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

    for speaker in SPEAKERS:
        if speaker == lowered:
            return speaker

    return DEFAULT_VOICE if DEFAULT_VOICE in SPEAKERS else "vivian"


model = None
model_sample_rate = DEFAULT_SAMPLE_RATE
model_tts_type = "unknown"
model_icl_capable = False
cancel_flags: dict[str, threading.Event] = {}
cancel_lock = threading.Lock()
# Serialize load / generate / unload so cancel+unload waits for the current generate.
model_lock = threading.RLock()
server_ready = False


def resolve_model_path(folder_name: str) -> str:
    full_path = os.path.join(MODELS_DIR, folder_name)
    if not os.path.isdir(full_path):
        raise FileNotFoundError(
            f"Model not found at {full_path}. "
            f"Download Base for ICL: "
            f"huggingface_hub.snapshot_download('mlx-community/{BASE_MODEL_FOLDER}', "
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


def detect_icl_capable(loaded_model) -> tuple[str, bool]:
    cfg = getattr(loaded_model, "config", None)
    tts_type = str(getattr(cfg, "tts_model_type", "base") or "base")
    tokenizer = getattr(loaded_model, "speech_tokenizer", None)
    has_encoder = bool(getattr(tokenizer, "has_encoder", False))
    # ICL is the Base path: ref_audio + ref_text + encoder.
    capable = tts_type == "base" and has_encoder
    return tts_type, capable


def enable_speech_tokenizer_encoder(loaded_model, model_path: str) -> bool:
    """Reload speech tokenizer WITH encoder weights.

    mlx_audio's post_load_hook currently sets encoder_config=None, which disables
    ICL even though Base model safetensors include encoder weights. Rebuild here.
    """
    import json
    from pathlib import Path

    import mlx.core as mx_local
    from mlx_audio.tts.models.qwen3_tts.config import (
        Qwen3TTSTokenizerConfig,
        Qwen3TTSTokenizerDecoderConfig,
        Qwen3TTSTokenizerEncoderConfig,
        filter_dict_for_dataclass,
    )
    from mlx_audio.tts.models.qwen3_tts.speech_tokenizer import Qwen3TTSSpeechTokenizer

    speech_tokenizer_path = Path(model_path) / "speech_tokenizer"
    config_path = speech_tokenizer_path / "config.json"
    if not config_path.is_file():
        print(f"[qwen-tts] No speech_tokenizer config at {config_path}")
        return False

    with open(config_path, encoding="utf-8") as f:
        tokenizer_config_dict = json.load(f)

    if not tokenizer_config_dict.get("encoder_config"):
        print("[qwen-tts] speech_tokenizer config has no encoder_config")
        return False

    decoder_config = None
    encoder_config = None
    if "decoder_config" in tokenizer_config_dict:
        filtered = filter_dict_for_dataclass(
            Qwen3TTSTokenizerDecoderConfig,
            tokenizer_config_dict["decoder_config"],
        )
        decoder_config = Qwen3TTSTokenizerDecoderConfig(**filtered)
    if "encoder_config" in tokenizer_config_dict:
        filtered = filter_dict_for_dataclass(
            Qwen3TTSTokenizerEncoderConfig,
            tokenizer_config_dict["encoder_config"],
        )
        encoder_config = Qwen3TTSTokenizerEncoderConfig(**filtered)

    tokenizer_config = Qwen3TTSTokenizerConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
    )
    for k, v in tokenizer_config_dict.items():
        if k not in ("decoder_config", "encoder_config") and hasattr(tokenizer_config, k):
            setattr(tokenizer_config, k, v)

    speech_tokenizer = Qwen3TTSSpeechTokenizer(tokenizer_config)
    tokenizer_weights = {}
    for wf in speech_tokenizer_path.glob("*.safetensors"):
        tokenizer_weights.update(mx_local.load(str(wf)))
    if not tokenizer_weights:
        print("[qwen-tts] No speech_tokenizer weights found")
        return False

    tokenizer_weights = Qwen3TTSSpeechTokenizer.sanitize(tokenizer_weights)
    speech_tokenizer.load_weights(list(tokenizer_weights.items()), strict=False)
    mx_local.eval(speech_tokenizer.parameters())
    speech_tokenizer.eval()

    if speech_tokenizer.encoder_model is not None:
        quantizer = speech_tokenizer.encoder_model.quantizer
        for layer in quantizer.rvq_first.vq.layers:
            layer.codebook.update_in_place()
        for layer in quantizer.rvq_rest.vq.layers:
            layer.codebook.update_in_place()

    if not speech_tokenizer.has_encoder:
        print("[qwen-tts] Encoder still missing after reload")
        return False

    loaded_model.load_speech_tokenizer(speech_tokenizer)
    print("[qwen-tts] Speech tokenizer encoder enabled for ICL voice cloning")
    return True


def ensure_model_loaded() -> None:
    """Load weights on first conversion request (caller must hold model_lock)."""
    global model, model_sample_rate, model_tts_type, model_icl_capable, DEFAULT_MODEL_FOLDER
    if model is not None:
        return

    DEFAULT_MODEL_FOLDER = resolve_default_model_folder()
    model_path = resolve_model_path(DEFAULT_MODEL_FOLDER)
    print(f"[qwen-tts] Loading model from {model_path} ...")
    loaded = load_model(model_path)

    tts_type, icl_capable = detect_icl_capable(loaded)
    if tts_type == "base" and not icl_capable:
        try:
            if enable_speech_tokenizer_encoder(loaded, model_path):
                tts_type, icl_capable = detect_icl_capable(loaded)
        except Exception as exc:
            print(f"[qwen-tts] Failed to enable ICL encoder: {exc}")

    model = loaded
    model_sample_rate = int(
        getattr(loaded, "sample_rate", DEFAULT_SAMPLE_RATE) or DEFAULT_SAMPLE_RATE
    )
    model_tts_type, model_icl_capable = tts_type, icl_capable
    print(
        f"[qwen-tts] Model loaded (sample_rate={model_sample_rate}, model={DEFAULT_MODEL_FOLDER}, "
        f"type={model_tts_type}, icl={model_icl_capable}, voice={DEFAULT_VOICE})"
    )
    if not model_icl_capable:
        print(
            "[qwen-tts] WARNING: Full ICL disabled. "
            "Will still pass preview ref_audio as x-vector anchor when available."
        )


def unload_model() -> bool:
    """Release model weights and MLX cache. Safe to call when already unloaded."""
    global model, model_tts_type, model_icl_capable
    with model_lock:
        if model is None:
            return False
        print("[qwen-tts] Unloading model...")
        model = None
        model_tts_type = "unknown"
        model_icl_capable = False
        gc.collect()
        if mx is not None:
            try:
                mx.clear_cache()
            except Exception as exc:
                print(f"[qwen-tts] mx.clear_cache failed: {exc}")
            try:
                metal = getattr(mx, "metal", None)
                if metal is not None and hasattr(metal, "clear_cache"):
                    metal.clear_cache()
            except Exception as exc:
                print(f"[qwen-tts] mx.metal.clear_cache failed: {exc}")
        print("[qwen-tts] Model unloaded.")
        return True


@contextmanager
def _hold_mlx_cache():
    """Skip mlx-audio's mid-generate mx.clear_cache().

    On M5, wiping the Metal cache between tokens/segments recycles Neural
    Accelerator buffers that still hold leftover data. M2 does not use those
    kernels, which is why the same project sounds fine there.
    """
    if mx is None or not hasattr(mx, "clear_cache"):
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
    language: str,
    instruct: str,
    temperature: float,
    ref_audio_path: Optional[str] = None,
    ref_text: Optional[str] = None,
    skip_icl: bool = False,
) -> tuple[np.ndarray, int, bool]:
    """Returns (audio, sample_rate, used_icl).

    Always keeps the selected `voice`. When a preview WAV+TXT exist, also pass
    them as ref_audio/ref_text so Base ICL (or x-vector) locks tone/energy.
    """
    assert model is not None
    # Keep newlines as pause cues (Node sends `\n\n\n` instead of `<break>` for Qwen).
    text = "\n".join(" ".join(line.split()) for line in text.split("\n")).strip(" \t")

    if skip_icl:
        # Bootstrap preview sample for this speaker id — do not clone from an old anchor.
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
    use_icl = bool(model_icl_capable and has_ref)

    # Base narration must clone the selected speaker's preview — never free-form.
    if not skip_icl and model_tts_type == "base" and not has_ref:
        raise ValueError(
            f"Missing voice preview anchor for '{voice}'. "
            f"Generate the preview first (expected under {VOICE_PREVIEW_DIR})."
        )

    print(
        f"[qwen-tts] synthesize voice={voice} icl={use_icl} has_ref={has_ref} "
        f"ref={ref_audio_path or '-'}"
    )

    # mlx-audio calls mx.clear_cache() after every codec token. On M5 that
    # recycles Neural Accelerator buffers with leftover data — first chunk OK,
    # the next one sounds destroyed. Hold the wipe until this generate finishes.
    with _hold_mlx_cache():
        if use_icl:
            # Full ICL: identity comes from the preview WAV of the selected voice.
            results = list(
                model.generate(
                    text=text,
                    voice=voice,
                    ref_audio=ref_audio_path,
                    ref_text=ref_text,
                    lang_code=language,
                    temperature=temperature,
                    verbose=False,
                )
            )
        elif has_ref and model_tts_type == "base":
            # Encoder missing: still pass ref_audio so speaker_encoder x-vector anchors tone.
            results = list(
                model.generate(
                    text=text,
                    voice=voice,
                    ref_audio=ref_audio_path,
                    ref_text=ref_text,
                    lang_code=language,
                    temperature=temperature,
                    verbose=False,
                )
            )
        else:
            gen_kwargs = dict(
                text=text,
                voice=voice,
                lang_code=language,
                temperature=temperature,
                verbose=False,
            )
            if model_tts_type == "custom_voice":
                gen_kwargs["instruct"] = instruct
            results = list(model.generate(**gen_kwargs))

    if not results:
        return np.zeros(0, dtype=np.float32), model_sample_rate, use_icl

    chunks = []
    sample_rate = model_sample_rate
    for result in results:
        audio = np.array(result.audio, dtype=np.float32).reshape(-1)
        if audio.size:
            chunks.append(audio)
        if getattr(result, "sample_rate", None):
            sample_rate = int(result.sample_rate)

    if not chunks:
        return np.zeros(0, dtype=np.float32), sample_rate, use_icl
    return np.concatenate(chunks), sample_rate, use_icl


def _mlx_version() -> str:
    try:
        from importlib.metadata import version

        return version("mlx")
    except Exception:
        return "?"


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global server_ready, DEFAULT_MODEL_FOLDER
    DEFAULT_MODEL_FOLDER = resolve_default_model_folder()
    # Default: stay lightweight at boot; weights load on the first /tts.
    # Opt in with QWEN_TTS_PRELOAD=1 for earlier ICL readiness.
    preload = _env_flag("QWEN_TTS_PRELOAD", default=False)
    if preload:
        try:
            with model_lock:
                ensure_model_loaded()
            print(
                f"[qwen-tts] Server ready (preloaded model={DEFAULT_MODEL_FOLDER}, "
                f"type={model_tts_type}, icl={model_icl_capable}, "
                f"mlx={_mlx_version()}, defaultVoice={DEFAULT_VOICE}, "
                f"previews={VOICE_PREVIEW_DIR})"
            )
        except Exception as exc:
            print(f"[qwen-tts] WARNING: model preload failed: {exc}")
            print("[qwen-tts] Server ready, but conversion will fail until the model loads.")
    else:
        print(
            f"[qwen-tts] Server ready (lazy model load; model={DEFAULT_MODEL_FOLDER}, "
            f"mlx={_mlx_version()}, defaultVoice={DEFAULT_VOICE}, "
            f"previews={VOICE_PREVIEW_DIR})"
        )
    server_ready = True
    yield
    server_ready = False
    unload_model()


app = FastAPI(title="Qwen3 TTS", lifespan=lifespan)


class TtsRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice: Optional[str] = None
    seed: Optional[int] = None  # accepted for Dia API compatibility; ignored
    language: Optional[str] = None
    instruct: Optional[str] = None
    temperature: Optional[float] = None
    refAudioPath: Optional[str] = None
    refText: Optional[str] = None
    # When true, force speaker-id generation (bootstrap preview) even if anchor exists.
    skipIcl: Optional[bool] = False
    jobId: Optional[str] = None


class CancelRequest(BaseModel):
    jobId: str


@app.get("/health")
def health():
    return {
        # Server process is up (app may start before weights are loaded).
        "ready": server_ready,
        "modelLoaded": model is not None,
        "provider": "qwen3-tts",
        "model": DEFAULT_MODEL_FOLDER,
        "ttsModelType": model_tts_type,
        "icl": model_icl_capable,
        "sampleRate": model_sample_rate,
        "defaultVoice": DEFAULT_VOICE,
        "defaultTemperature": DEFAULT_TEMPERATURE,
        "previewDir": VOICE_PREVIEW_DIR,
        "speakers": sorted(SPEAKERS),
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
        # Do not unload on cancel — keeps ICL ready for the next chunk/job.
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

    # Hold lock for load + generate so /tts/unload waits until this request finishes.
    with model_lock:
        try:
            ensure_model_loaded()
        except Exception as exc:
            if job_id:
                with cancel_lock:
                    cancel_flags.pop(job_id, None)
            raise HTTPException(
                status_code=503, detail=f"Failed to load TTS model: {exc}"
            ) from exc

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
            # Keep model loaded; only unload on explicit /tts/unload.
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
    parser = argparse.ArgumentParser(description="Qwen3-TTS HTTP server for AuraReader")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
