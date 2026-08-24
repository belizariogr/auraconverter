"""
HTTP TTS server wrapping Qwen3-TTS (MLX) for AuraReader.

Prefers the Pro Base model (1.7B 8-bit) so narration can use ICL voice cloning with a
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
BASE_MODEL_FOLDER = "Qwen3-TTS-12Hz-1.7B-Base-8bit"
CUSTOM_MODEL_FOLDER = "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
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

LANGUAGE_MAP = {
    "auto": "Auto",
    "en": "English",
    "en-us": "English",
    "en-gb": "English",
    "english": "English",
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-hans": "Chinese",
    "chinese": "Chinese",
    "ja": "Japanese",
    "ja-jp": "Japanese",
    "japanese": "Japanese",
    "ko": "Korean",
    "ko-kr": "Korean",
    "korean": "Korean",
    "de": "German",
    "de-de": "German",
    "german": "German",
    "fr": "French",
    "fr-fr": "French",
    "french": "French",
    "ru": "Russian",
    "ru-ru": "Russian",
    "russian": "Russian",
    "pt": "Portuguese",
    "pt-br": "Portuguese",
    "portuguese": "Portuguese",
    "es": "Spanish",
    "es-es": "Spanish",
    "spanish": "Spanish",
    "it": "Italian",
    "it-it": "Italian",
    "italian": "Italian",
}


def resolve_language(language: Optional[str]) -> str:
    raw = (language or DEFAULT_LANGUAGE).strip() or DEFAULT_LANGUAGE
    mapped = LANGUAGE_MAP.get(raw.lower().replace("_", "-"))
    if mapped:
        return mapped
    if raw[:1].isupper():
        return raw
    return raw.title()
# Official Qwen3-TTS / mlx-audio hard defaults (generate_config / qwen3_tts.py).
# Low temperature (e.g. 0.3) or greedy (0) degenerates codec tokens → silence, beeps, dropouts.
DEFAULT_TEMPERATURE = float(os.environ.get("QWEN_TTS_TEMPERATURE", "0.9"))
DEFAULT_TOP_K = int(os.environ.get("QWEN_TTS_TOP_K", "50"))
DEFAULT_TOP_P = float(os.environ.get("QWEN_TTS_TOP_P", "1.0"))
DEFAULT_REPETITION_PENALTY = float(os.environ.get("QWEN_TTS_REPETITION_PENALTY", "1.05"))
# Official hard default is 2048; mlx-audio's generate() default is 4096.
DEFAULT_MAX_TOKENS = int(os.environ.get("QWEN_TTS_MAX_TOKENS", "2048"))
DEFAULT_INSTRUCT = os.environ.get(
    "QWEN_TTS_INSTRUCT",
    (
        "Speak as a warm audiobook narrator at a normal conversational volume. "
        "Use a clear, natural speaking voice with light expressive emotion. "
        "Do not whisper, murmur, or sound breathy or hushed. "
        "Do not sound flat, robotic, overly dramatic, or theatrical. "
        "Keep a steady storytelling pace suitable for book narration."
    ),
)
DEFAULT_REF_TEXT = os.environ.get(
    "QWEN_TTS_PREVIEW_TEXT",
    (
        "This is a preview of my narration voice. "
        "I am reading at a normal volume, clearly and with a little warmth."
    ),
)
VOICE_PREVIEW_DIR = os.environ.get(
    "VOICE_PREVIEW_DIR",
    os.path.join(REPO_ROOT, "assets", "voice-previews"),
)
# Must match Node `VOICE_PREVIEW_CACHE_VERSION` (server.ts).
VOICE_PREVIEW_CACHE_VERSION = os.environ.get(
    "QWEN_TTS_PREVIEW_CACHE_VERSION", "1.7b-narrate-v1"
)

# Qwen lang_code → Node voicePreviewCacheTag used in disk filenames.
QWEN_LANG_TO_PREVIEW_TAG = {
    "Auto": "auto",
    "Portuguese": "pt-br",
    "English": "en-us",
    "Spanish": "es-es",
    "French": "fr-fr",
    "Italian": "it-it",
    "German": "de-de",
    "Japanese": "ja-jp",
    "Korean": "ko-kr",
    "Chinese": "zh-cn",
    "Russian": "ru-ru",
}

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

# mlx-audio CustomVoice expects Title_Case speaker ids.
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


def speaker_api_name(voice: str) -> str:
    key = resolve_voice(voice)
    return SPEAKER_API_NAMES.get(key, key.replace("_", " ").title().replace(" ", "_"))


def model_folder_present(folder_name: str) -> bool:
    return os.path.isdir(os.path.join(MODELS_DIR, folder_name))


base_model = None
custom_model = None
active_kind: Optional[str] = None
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


def preview_locale_tag(language: Optional[str] = None) -> Optional[str]:
    """Map Qwen language name / BCP-47 to the Node preview filename tag."""
    raw = (language or "").strip()
    if not raw:
        return None

    mapped_lang = LANGUAGE_MAP.get(raw.lower().replace("_", "-"))
    qwen_name = mapped_lang or (raw if raw[:1].isupper() else raw.title())
    tag = QWEN_LANG_TO_PREVIEW_TAG.get(qwen_name)
    if tag:
        return tag

    # Already a locale-like tag (pt-BR → pt-br).
    lowered = raw.lower().replace("_", "-")
    if "-" in lowered or lowered in ("auto",):
        return lowered

    return None


def preview_paths_for_voice(
    voice: str,
    language: Optional[str] = None,
) -> tuple[str, str]:
    """Resolve WAV+TXT paths. Prefer voice_locale_sNNN_version (Node), then legacy."""
    safe = resolve_voice(voice).replace(" ", "_").lower()
    locale = preview_locale_tag(language)
    keys: list[str] = []
    if locale:
        # Default-speed tag used when Node does not pass an explicit ref path.
        keys.append(f"{safe}_{locale}_s100_{VOICE_PREVIEW_CACHE_VERSION}")
        keys.append(f"{safe}_{locale}_{VOICE_PREVIEW_CACHE_VERSION}")
    keys.append(f"{safe}_{VOICE_PREVIEW_CACHE_VERSION}")

    for key in keys:
        wav = os.path.join(VOICE_PREVIEW_DIR, f"{key}.wav")
        txt = os.path.join(VOICE_PREVIEW_DIR, f"{key}.txt")
        if os.path.isfile(wav) and os.path.isfile(txt):
            return wav, txt

    # Default to locale-aware path even if missing (caller reports the expected name).
    preferred = keys[0]
    return (
        os.path.join(VOICE_PREVIEW_DIR, f"{preferred}.wav"),
        os.path.join(VOICE_PREVIEW_DIR, f"{preferred}.txt"),
    )


def load_preview_anchor(
    voice: str,
    language: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    wav_path, txt_path = preview_paths_for_voice(voice, language)
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


def _drop_mlx_model(kind: str) -> None:
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
    if mx is not None:
        try:
            mx.clear_cache()
        except Exception:
            pass


def _load_mlx_model(folder_name: str):
    model_path = resolve_model_path(folder_name)
    print(f"[qwen-tts] Loading {folder_name} from {model_path} ...")
    loaded = load_model(model_path)
    sample_rate = int(
        getattr(loaded, "sample_rate", DEFAULT_SAMPLE_RATE) or DEFAULT_SAMPLE_RATE
    )
    if folder_name == BASE_MODEL_FOLDER:
        tts_type, icl_capable = detect_icl_capable(loaded)
        if tts_type == "base" and not icl_capable:
            try:
                if enable_speech_tokenizer_encoder(loaded, model_path):
                    tts_type, icl_capable = detect_icl_capable(loaded)
            except Exception as exc:
                print(f"[qwen-tts] Failed to enable ICL encoder: {exc}")
        return loaded, sample_rate, tts_type, icl_capable
    return loaded, sample_rate, "custom_voice", False


def ensure_model_loaded(kind: Optional[str] = None) -> None:
    """Load Base (ICL) or CustomVoice (speaker presets). Caller holds model_lock."""
    global base_model, custom_model, active_kind, model_sample_rate
    global model_tts_type, model_icl_capable, DEFAULT_MODEL_FOLDER

    if kind is None:
        kind = "base" if model_folder_present(BASE_MODEL_FOLDER) else "custom"

    if kind == "base":
        if base_model is not None:
            active_kind = "base"
            model_tts_type = "base"
            model_icl_capable = bool(
                getattr(getattr(base_model, "speech_tokenizer", None), "has_encoder", False)
            )
            DEFAULT_MODEL_FOLDER = BASE_MODEL_FOLDER
            return
        if custom_model is not None:
            print("[qwen-tts] Unloading CustomVoice to free memory for Base")
            _drop_mlx_model("custom")
        loaded, sample_rate, tts_type, icl_capable = _load_mlx_model(BASE_MODEL_FOLDER)
        base_model = loaded
        active_kind = "base"
        model_sample_rate = sample_rate
        model_tts_type = tts_type
        model_icl_capable = icl_capable
        DEFAULT_MODEL_FOLDER = BASE_MODEL_FOLDER
        print(
            f"[qwen-tts] Base ready (sample_rate={model_sample_rate}, "
            f"icl={model_icl_capable})"
        )
        return

    if kind == "custom":
        if custom_model is not None:
            active_kind = "custom"
            model_tts_type = "custom_voice"
            model_icl_capable = False
            DEFAULT_MODEL_FOLDER = CUSTOM_MODEL_FOLDER
            return
        if base_model is not None:
            print("[qwen-tts] Unloading Base to free memory for CustomVoice")
            _drop_mlx_model("base")
        loaded, sample_rate, tts_type, icl_capable = _load_mlx_model(CUSTOM_MODEL_FOLDER)
        custom_model = loaded
        active_kind = "custom"
        model_sample_rate = sample_rate
        model_tts_type = tts_type
        model_icl_capable = icl_capable
        DEFAULT_MODEL_FOLDER = CUSTOM_MODEL_FOLDER
        print(
            f"[qwen-tts] CustomVoice ready (sample_rate={model_sample_rate}, "
            f"speaker presets enabled)"
        )
        return

    raise ValueError(f"Unknown model kind: {kind}")


def unload_model() -> bool:
    """Release model weights and MLX cache. Safe to call when already unloaded."""
    global model_sample_rate
    with model_lock:
        had = base_model is not None or custom_model is not None
        if not had:
            return False
        print("[qwen-tts] Unloading model(s)...")
        _drop_mlx_model("base")
        _drop_mlx_model("custom")
        model_sample_rate = DEFAULT_SAMPLE_RATE
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


def _audio_from_results(
    results: list,
    default_sample_rate: int,
) -> tuple[np.ndarray, int]:
    if not results:
        return np.zeros(0, dtype=np.float32), default_sample_rate

    chunks: list[np.ndarray] = []
    sample_rate = default_sample_rate
    for result in results:
        audio = np.array(result.audio, dtype=np.float32).reshape(-1)
        if audio.size:
            chunks.append(audio)
        if getattr(result, "sample_rate", None):
            sample_rate = int(result.sample_rate)

    if not chunks:
        return np.zeros(0, dtype=np.float32), sample_rate

    return np.concatenate(chunks), sample_rate


def synthesize(
    text: str,
    voice: str,
    language: str,
    instruct: str,
    temperature: float,
    top_k: int = DEFAULT_TOP_K,
    top_p: float = DEFAULT_TOP_P,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    ref_audio_path: Optional[str] = None,
    ref_text: Optional[str] = None,
    skip_icl: bool = False,
) -> tuple[np.ndarray, int, bool]:
    """Returns (audio, sample_rate, used_icl).

    Preview bootstrap (skipIcl) uses CustomVoice speaker presets so each voice
    sounds distinct. Narration clones the saved preview WAV via Base ICL.
    """
    # Keep newlines as pause cues (Node sends `\n\n\n` instead of `<break>` for Qwen).
    text = "\n".join(" ".join(line.split()) for line in text.split("\n")).strip(" \t")

    if skip_icl:
        ref_audio_path = None
        ref_text = None
    else:
        if not ref_audio_path or not ref_text:
            auto_wav, auto_txt = load_preview_anchor(voice, language)
            if not ref_audio_path:
                ref_audio_path = auto_wav
            if not ref_text:
                ref_text = auto_txt

    has_ref = bool(
        ref_audio_path and ref_text and os.path.isfile(ref_audio_path)
    )
    use_icl = bool(has_ref and not skip_icl and model_folder_present(BASE_MODEL_FOLDER))

    if not skip_icl and not has_ref and model_folder_present(BASE_MODEL_FOLDER):
        raise ValueError(
            f"Missing voice preview anchor for '{voice}'. "
            f"Generate the preview first (expected under {VOICE_PREVIEW_DIR})."
        )

    print(
        f"[qwen-tts] synthesize voice={voice} icl={use_icl} has_ref={has_ref} "
        f"ref={ref_audio_path or '-'} temp={temperature} top_k={top_k} "
        f"top_p={top_p} rep={repetition_penalty} max_tokens={max_tokens}"
    )

    sampling = dict(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
        verbose=False,
    )

    with _hold_mlx_cache():
        if use_icl:
            ensure_model_loaded("base")
            assert base_model is not None
            results = list(
                base_model.generate(
                    text=text,
                    ref_audio=ref_audio_path,
                    ref_text=ref_text,
                    lang_code=language,
                    **sampling,
                )
            )
            audio, sample_rate = _audio_from_results(results, model_sample_rate)
            return audio, sample_rate, True

        if not model_folder_present(CUSTOM_MODEL_FOLDER):
            raise ValueError(
                f"CustomVoice model not found at {os.path.join(MODELS_DIR, CUSTOM_MODEL_FOLDER)}. "
                "Install it to generate distinct voice previews."
            )

        ensure_model_loaded("custom")
        assert custom_model is not None
        speaker = speaker_api_name(voice)
        print(f"[qwen-tts] synthesize speaker={speaker} custom_voice preview/narration")
        gen_kwargs = dict(
            text=text,
            voice=speaker,
            lang_code=language,
            **sampling,
        )
        if instruct:
            gen_kwargs["instruct"] = instruct

        results = list(custom_model.generate(**gen_kwargs))
        audio, sample_rate = _audio_from_results(results, model_sample_rate)
        return audio, sample_rate, False


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
    topK: Optional[int] = None
    topP: Optional[float] = None
    repetitionPenalty: Optional[float] = None
    maxTokens: Optional[int] = None
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
        "modelLoaded": base_model is not None or custom_model is not None,
        "baseLoaded": base_model is not None,
        "customLoaded": custom_model is not None,
        "activeKind": active_kind,
        "provider": "qwen3-tts",
        "model": DEFAULT_MODEL_FOLDER,
        "ttsModelType": model_tts_type,
        "icl": model_icl_capable,
        "sampleRate": model_sample_rate,
        "defaultVoice": DEFAULT_VOICE,
        "defaultTemperature": DEFAULT_TEMPERATURE,
        "defaultTopK": DEFAULT_TOP_K,
        "defaultTopP": DEFAULT_TOP_P,
        "defaultRepetitionPenalty": DEFAULT_REPETITION_PENALTY,
        "defaultMaxTokens": DEFAULT_MAX_TOKENS,
        "previewDir": VOICE_PREVIEW_DIR,
        "speakers": sorted(SPEAKERS),
    }


@app.post("/tts")
def tts(req: TtsRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    voice = resolve_voice(req.voice)
    language = resolve_language(req.language)
    instruct = (req.instruct or DEFAULT_INSTRUCT).strip() or DEFAULT_INSTRUCT
    temperature = (
        DEFAULT_TEMPERATURE if req.temperature is None else float(req.temperature)
    )
    if temperature < 0:
        temperature = 0.0
    if temperature > 1.5:
        temperature = 1.5

    top_k = DEFAULT_TOP_K if req.topK is None else int(req.topK)
    if top_k < 0:
        top_k = 0

    top_p = DEFAULT_TOP_P if req.topP is None else float(req.topP)
    if top_p < 0:
        top_p = 0.0
    if top_p > 1.0:
        top_p = 1.0

    repetition_penalty = (
        DEFAULT_REPETITION_PENALTY
        if req.repetitionPenalty is None
        else float(req.repetitionPenalty)
    )
    if repetition_penalty < 1.0:
        repetition_penalty = 1.0

    max_tokens = DEFAULT_MAX_TOKENS if req.maxTokens is None else int(req.maxTokens)
    if max_tokens < 1:
        max_tokens = 1

    ref_audio_path = (req.refAudioPath or "").strip() or None
    ref_text = (req.refText or "").strip() or None
    skip_icl = bool(req.skipIcl)
    job_id = req.jobId

    print(
        f"[qwen-tts] /tts request voice={voice!r} (from {req.voice!r}) "
        f"language={language!r} skipIcl={skip_icl} refAudioPath={ref_audio_path or '-'} "
        f"jobId={job_id or '-'} temp={temperature}"
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
            audio, sample_rate, used_icl = synthesize(
                text,
                voice,
                language,
                instruct,
                temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_tokens=max_tokens,
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
        except FileNotFoundError as exc:
            if job_id:
                with cancel_lock:
                    cancel_flags.pop(job_id, None)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
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
