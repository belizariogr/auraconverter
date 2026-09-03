"""Servidor HTTP local para o Breeze TTS 2 em Apple Silicon/MLX."""

from __future__ import annotations

import argparse
import base64
import gc
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import uvicorn
from breeze_tts_mlx.runtime import BreezeMLXRuntime, MLXRuntimeConfig
from breeze_tts_mlx.sampling import SamplingConfig
from breeze_tts_mlx.templates import get_template, prepare_inputs
from fastapi import FastAPI
from pydantic import BaseModel

DEFAULT_SAMPLE_RATE = 24_000
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = Path(os.environ.get("BREEZE_TTS_MODELS_DIR", SCRIPT_DIR / "models"))
MODEL_FOLDER = "Breeze-TTS-2-mlx"
PORT = int(os.environ.get("BREEZE_TTS_PORT", os.environ.get("TTS_PORT", "8765")))
HOST = os.environ.get("BREEZE_TTS_HOST", "127.0.0.1")

model: Optional[BreezeMLXRuntime] = None
model_lock = threading.RLock()
cancelled_jobs: set[str] = set()

VOICE_INSTRUCTIONS = {
    "vivian": "A warm, clear young woman with a calm and intimate audiobook delivery.",
    "serena": "A gentle woman with a soft, reassuring voice and patient audiobook delivery.",
    "sohee": "An expressive woman with bright diction and subtle emotional nuance.",
    "ono_anna": "A confident woman with precise diction and a natural conversational rhythm.",
    "ryan": "A calm man with a steady, warm low register suited to long audiobook chapters.",
    "aiden": "An engaging man with friendly energy and a clear storytelling voice.",
    "eric": "A mature man with formal diction, composed pacing, and quiet authority.",
    "dylan": "A grounded man with a rich, measured voice and restrained emotion.",
    "uncle_fu": "A mature man with a deep, patient, grandfatherly storytelling voice.",
}


class TtsRequest(BaseModel):
    text: str
    voice: str = "breeze_narrator"
    jobId: Optional[str] = None
    instruct: str = "Speak warmly and clearly as an audiobook narrator."
    refAudioPath: Optional[str] = None
    refText: Optional[str] = None
    skipIcl: bool = False
    temperature: float = 0.9
    topK: int = 50
    topP: float = 1.0
    repetitionPenalty: float = 1.1
    maxTokens: int = 1500


class CancelRequest(BaseModel):
    jobId: str


def model_path() -> Path:
    path = MODELS_DIR / MODEL_FOLDER
    if not path.is_dir():
        raise FileNotFoundError(f"Modelo Breeze não encontrado em {path}.")
    return path


def ensure_model() -> BreezeMLXRuntime:
    global model
    with model_lock:
        if model is None:
            model = BreezeMLXRuntime(model_path(), audio_device="auto")
        return model


def unload_model() -> bool:
    global model
    with model_lock:
        if model is None:
            return False
        model = None
        gc.collect()
        return True


def encode_pcm(audio: np.ndarray) -> str:
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not samples.size:
        return ""
    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(samples)))
    if peak > 1.0:
        samples *= np.float32(0.999 / peak)
    pcm = np.rint(samples * 32767.0).astype("<i2", copy=False)
    return base64.b64encode(pcm.tobytes()).decode("ascii")


def resolve_template(request: TtsRequest) -> tuple[str, dict[str, str]]:
    has_reference = bool(request.refAudioPath and request.refText)
    if has_reference and not request.skipIcl:
        return "ref_edit_tata", {
            "ref_audio_path": str(request.refAudioPath),
            "ref_text": str(request.refText).strip(),
        }
    return "tts_instruction", {}


def voice_instruction(request: TtsRequest) -> str:
    key = request.voice.strip().lower().replace(" ", "_")
    preset = VOICE_INSTRUCTIONS.get(key, VOICE_INSTRUCTIONS["vivian"])
    instruction = request.instruct.strip()
    return f"{preset} {instruction}".strip()


def synthesize(request: TtsRequest) -> tuple[np.ndarray, int, bool]:
    if not request.text.strip():
        raise ValueError("O texto para o Breeze não pode estar vazio.")
    if bool(request.refAudioPath) != bool(request.refText):
        raise ValueError("refAudioPath e refText devem ser informados juntos.")

    runtime = ensure_model()
    template_name, reference = resolve_template(request)
    payload: dict[str, str] = {
        "id": request.jobId or "aura-request",
        "text": request.text,
        "speaker": "S0",
        "instruction": voice_instruction(request),
        **reference,
    }
    inputs = prepare_inputs(
        runtime.tokenizer,
        runtime.audio_tokenizer,
        runtime,
        [payload],
        get_template(template_name),
        guidance_scale=1.0,
        guidance_scale_ref=None,
        guidance_scale_ins=None,
    )
    sampling = SamplingConfig(
        temperature=request.temperature,
        top_k=request.topK,
        top_p=request.topP,
        do_sample=True,
    )
    runtime.runtime_config = MLXRuntimeConfig(
        max_new_tokens=request.maxTokens,
        max_seq_len=2048,
        repetition_penalty=request.repetitionPenalty,
        codec_chunk_frames=2,
        backbone_sampling=sampling,
        depth_sampling=sampling,
    )
    audio_parts: list[np.ndarray] = []
    for chunk in runtime.iter_audio_chunks(inputs, request_id=request.jobId or "aura-request"):
        if request.jobId and request.jobId in cancelled_jobs:
            return np.zeros(0, dtype=np.float32), runtime.sample_rate, False
        audio_parts.append(chunk.audio)
    audio = np.concatenate(audio_parts) if audio_parts else np.zeros(0, dtype=np.float32)
    return audio, runtime.sample_rate, bool(reference)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    print(f"[breeze] Server ready (lazy model load; model={model_path()})", flush=True)
    yield
    unload_model()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "ready": True,
        "provider": "breeze",
        "model": MODEL_FOLDER,
        "modelDir": str(MODELS_DIR / MODEL_FOLDER),
        "sampleRate": model.sample_rate if model else DEFAULT_SAMPLE_RATE,
        "modelLoaded": model is not None,
        "voices": ["breeze_narrator"],
    }


@app.post("/tts")
def tts(request: TtsRequest) -> dict[str, object]:
    try:
        audio, sample_rate, used_reference = synthesize(request)
        cancelled = bool(request.jobId and request.jobId in cancelled_jobs)
        if request.jobId:
            cancelled_jobs.discard(request.jobId)
        return {
            "sampleRate": sample_rate,
            "audioData": "" if cancelled else encode_pcm(audio),
            "format": "pcm_s16le",
            "cancelled": cancelled,
            "icl": used_reference,
            "voice": request.voice,
        }
    except Exception as exc:
        raise RuntimeError(f"Falha no Breeze TTS: {exc}") from exc


@app.post("/tts/cancel")
def cancel(request: CancelRequest) -> dict[str, bool]:
    cancelled_jobs.add(request.jobId)
    return {"ok": True}


@app.post("/tts/unload")
def unload() -> dict[str, bool]:
    return {"unloaded": unload_model()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
