from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch


def require_sox() -> str:
    """Return the SoX executable or fail before qwen_tts prints a cryptic error."""
    executable = shutil.which("sox")
    if executable is None:
        raise RuntimeError(
            "The Qwen audio tokenizer requires the native SoX executable. "
            "Install it with `brew install sox`, then confirm `sox --version` works. "
            "On Apple Silicon, ensure /opt/homebrew/bin is on PATH."
        )
    return executable


def resolve_audio_device(requested: str, *, dtype: torch.dtype) -> torch.device:
    if requested == "auto":
        requested = "mps" if torch.backends.mps.is_available() else "cpu"
    if requested not in {"cpu", "mps"}:
        raise ValueError("audio device must be one of: auto, cpu, mps")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS was requested for the audio tokenizer but is unavailable"
        )
    device = torch.device(requested)
    if dtype == torch.float16 and device.type != "mps":
        raise RuntimeError(
            "Run an FP16 codec on MPS; use an Apple-Silicon Mac with "
            "--audio-device auto or mps"
        )
    return device


def _extract_audio(decoded: Any) -> np.ndarray:
    if hasattr(decoded, "audio_values"):
        decoded = decoded.audio_values
    while isinstance(decoded, (list, tuple)):
        decoded = decoded[0]
    if isinstance(decoded, np.ndarray):
        audio = decoded
    elif isinstance(decoded, torch.Tensor):
        audio = decoded.detach().float().cpu().numpy()
    else:
        raise TypeError(f"Unsupported decoded audio type: {type(decoded)!r}")
    while audio.ndim > 1:
        audio = audio[0]
    return np.ascontiguousarray(audio, dtype=np.float32)


class AudioTokenizer:
    """Qwen codec with an explicit FP32 or FP16 execution policy."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: str = "auto",
        dtype: str = "float32",
    ) -> None:
        require_sox()
        from qwen_tts import Qwen3TTSTokenizer

        if dtype not in {"float16", "float32"}:
            raise ValueError("audio tokenizer dtype must be float16 or float32")
        self.dtype = torch.float16 if dtype == "float16" else torch.float32
        self.device = resolve_audio_device(device, dtype=self.dtype)
        self.model_dir = Path(model_dir)
        if not self.model_dir.is_dir():
            raise FileNotFoundError(
                f"Missing bundled audio tokenizer: {self.model_dir}"
            )
        self.tokenizer = Qwen3TTSTokenizer.from_pretrained(
            str(self.model_dir), device_map=str(self.device), dtype=self.dtype
        )
        if self.tokenizer.model is None:
            raise RuntimeError("Qwen audio tokenizer loaded without its neural model")
        self.tokenizer.model.to(device=self.device, dtype=self.dtype).eval()
        for parameter in self.tokenizer.model.parameters():
            parameter.requires_grad_(False)
            if parameter.is_floating_point() and parameter.dtype != self.dtype:
                raise RuntimeError(
                    f"audio tokenizer must remain {self.dtype}, found "
                    f"parameter dtype {parameter.dtype}"
                )
        self._stream_runtimes: dict[int, Any] = {}

    @property
    def model(self) -> Any:
        return self.tokenizer.model

    @property
    def sample_rate(self) -> int:
        config = self.tokenizer.model.decoder.config
        return int(getattr(config, "sampling_rate", 24000))

    def encode(self, *args: Any, **kwargs: Any) -> Any:
        with torch.inference_mode():
            return self.tokenizer.encode(*args, **kwargs)

    def stream_runtime(self, chunk_frames: int) -> Any:
        runtime = self._stream_runtimes.get(chunk_frames)
        if runtime is not None:
            return runtime
        from .codec_stream.stream.runtime import (
            MultiRequestStreamRuntime,
            QwenStreamRuntimeConfig,
        )

        runtime = MultiRequestStreamRuntime(
            self.tokenizer,
            QwenStreamRuntimeConfig(
                chunk_frames=chunk_frames,
                non_integer_chunk_strategy="eager",
                num_lanes=1,
                max_active_reqs=1,
                fast=False,
                lifecycle_assert_mode="raise",
                device=self.device,
                dtype=self.dtype,
            ),
        )
        self._stream_runtimes[chunk_frames] = runtime
        return runtime

    def decode_chunk(
        self,
        runtime: Any,
        request_id: str,
        frames: list[np.ndarray],
        *,
        reset: bool,
    ) -> np.ndarray:
        frame_array = np.stack(frames, axis=0).astype(np.int64, copy=False)
        codes = torch.from_numpy(frame_array.T[None]).to(
            device=self.device, dtype=torch.long
        )
        with torch.inference_mode():
            decoded = runtime.decode_request_chunk(request_id, codes, reset=reset)
        return _extract_audio(decoded)
