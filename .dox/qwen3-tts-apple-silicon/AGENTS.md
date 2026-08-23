## Purpose

Qwen3-TTS FastAPI server on Apple Silicon (MLX). Used by `server.ts` on darwin.

## Ownership

- `tts_server.py` — `/health`, `/tts`, cancel, unload
- `models/` — 0.6B Base-8bit (ICL) or CustomVoice-8bit
- `.venv` — mlx / mlx-audio

## Local Contracts

- During `model.generate()`, patch `mx.clear_cache` to no-op; restore and clear once after the request (M5 quality).
- Pin **mlx / mlx-metal ≥ 0.32.1**. 0.30.3 on M5 drops the middle of Qwen3-TTS audio ([mlx-audio#464](https://github.com/Blaizzy/mlx-audio/issues/464)); NAX kernel fixes landed after that.
- Do **not** set `MLX_ENABLE_TF32=0`.
- Node splits Qwen prompts to ~280 chars; Kokoro may pack 5 paragraphs.
- Do **not** insert `<break>` into extracted text when the engine is Qwen. Node replaces those lines with `\n\n\n`. Collapse intra-line whitespace only; keep newlines.
- CustomVoice `generate()` ignores `split_pattern`; Node owns text chunking.
- Preview WAV+TXT under `assets/voice-previews/` are ICL anchors for Base.

## Work Guidance

- Do not call `mx.clear_cache()` between codec tokens (mlx-audio does; we hold it).
- Keep API fields compatible with Kokoro/Torch servers (`refAudioPath`, `skipIcl`, `jobId`).
- `/tts` `language` is required for narration: map BCP-47 / aliases to Qwen names (`Portuguese`, `English`, `Auto`, …).

## Verification

- After edits: fully quit Electron so Python reloads this file.

## Child DOX Index

(none)
