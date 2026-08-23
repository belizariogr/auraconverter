## Purpose

Kokoro TTS: ONNX Runtime (`tts_server.py`) and MLX (`tts_server_mlx.py`).

## Ownership

- `tts_server_mlx.py` — Apple Silicon; weights `Kokoro-82M-bf16`
- `tts_server.py` — ONNX (+ CUDA / CoreML / DirectML / CPU)

## Local Contracts

- Drive `generate(..., split_pattern=None)` and hold `mx.clear_cache` for the whole request. mlx-audio otherwise splits on `\\n+` and wipes Metal between windows (M5: first window OK, rest harsh).
- Do not force `MLX_ENABLE_TF32=0` (M5 Neural Accelerators; large slowdown).
- Ignore ICL fields.

## Work Guidance

- Prefer local `voices/*.safetensors` when present.
- `KOKORO_DEVICE=cpu` is the diagnostic escape hatch on Metal bugs.
- Map request `language` to MLX `lang_code` (`p`, `a`, …) or ONNX `lang` (`pt-br`, `en-us`, …).

## Verification

- Full app quit after Python edits.

## Child DOX Index

(none)
