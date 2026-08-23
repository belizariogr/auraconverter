## Purpose

Qwen3-TTS FastAPI server on Apple Silicon (MLX). Used by `server.ts` on darwin.

## Ownership

- `tts_server.py` — `/health`, `/tts`, cancel, unload
- `models/` — 0.6B Base-8bit (ICL) or CustomVoice-8bit
- `.venv` — mlx / mlx-audio

## Local Contracts

- During `model.generate()`, patch `mx.clear_cache` to no-op; restore and clear once after the request. That is the M5 quality fix (mlx-audio otherwise wipes Metal between tokens).
- Do **not** set `MLX_ENABLE_TF32=0`. TF32/NAX is how M5 stays fast; the cache hold is enough to avoid dirty-buffer audio.
- CustomVoice `generate()` ignores `split_pattern`; Node owns text chunking.
- Preview WAV+TXT under `assets/voice-previews/` are ICL anchors for Base.

## Work Guidance

- Do not call `mx.clear_cache()` between codec tokens (mlx-audio does; we hold it).
- Keep API fields compatible with Kokoro/Torch servers (`refAudioPath`, `skipIcl`, `jobId`).

## Verification

- After edits: fully quit Electron so Python reloads this file.

## Child DOX Index

(none)
