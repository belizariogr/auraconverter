## Purpose

Qwen3-TTS FastAPI server on Apple Silicon (MLX). Used by `server.ts` on darwin.

## Ownership

- `tts_server.py` — `/health`, `/tts`, cancel, unload
- `models/` — **1.7B** Base-8bit (ICL) and CustomVoice-8bit (mlx-community)
- `.venv` — mlx / mlx-audio

## Local Contracts

- During `model.generate()`, patch `mx.clear_cache` to no-op; restore and clear once after the request (M5 quality).
- Pin **mlx / mlx-metal ≥ 0.32.1** and **mlx-audio ≥ 0.5.0** (PyPI). 0.30.3 on M5 drops mid-utterance Qwen audio ([mlx-audio#464](https://github.com/Blaizzy/mlx-audio/issues/464)); NAX fixes and Qwen float32 upcast fix landed in later releases.
- Do **not** set `MLX_ENABLE_TF32=0`.
- Sampling defaults (official Qwen3-TTS / mlx-audio): `temperature=0.9`, `top_k=50`, `top_p=1.0`, `repetition_penalty=1.05`, `max_tokens=2048`. Pass them explicitly on every `generate()`. Low temperature degenerates codec tokens.
- Node splits Qwen prompts to ~280 chars; Kokoro may pack 5 paragraphs.
- Do **not** insert `<break>` into extracted text when the engine is Qwen. Node replaces those lines with `\n\n\n`. Collapse intra-line whitespace only; keep newlines.
- CustomVoice `generate()` ignores `split_pattern`; Node owns text chunking.
- Preview WAV+TXT under `AURA_DATA_DIR/assets/voice-previews/` (runtime-generated; not git / not packaged) are ICL anchors for Base (one per voice+locale+speed). Filename: `{voice}_{locale}_{sNNN}_{QWEN_TTS_PREVIEW_CACHE_VERSION}` (e.g. `vivian_pt-br_s100_1.7b-narrate-v1`). Current version default: `1.7b-narrate-v1`. Speed ≠ 1× is applied with ffmpeg `atempo` after CustomVoice generate, then saved — narrate uses that same file as `ref_audio`.
- **Preview bootstrap (`skipIcl`)** must use **CustomVoice** speaker presets — Base has no built-in voices and would make every preview identical.
- Narration with Base loads the saved preview WAV (matching language + speed) and clones via ICL (`ref_audio` + `ref_text`; do not pass `voice` on that path). ICL **ignores** `instruct` — delivery style (normal volume, light emotion, no whisper) must be baked into the preview WAV via CustomVoice `instruct` + `previewText`. Do **not** apply encode-time `atempo` on top — speed already lives in the ICL preview.
- Fallback `load_preview_anchor(voice, language)` resolves locale-aware paths (`Portuguese` → `pt-br`); legacy `voice_version` names still accepted.
- PCM encode path: float→int16 with clip only if peak > 1.0; no loudness/EQ. MP3/M4B encode at native 24 kHz (no aresample).

## Work Guidance

- Do not call `mx.clear_cache()` between codec tokens (mlx-audio does; we hold it).
- Keep API fields compatible with Kokoro/Torch servers (`refAudioPath`, `skipIcl`, `jobId`, optional `topK`/`topP`/`repetitionPenalty`/`maxTokens`).
- `/tts` `language` is required for narration: map BCP-47 / aliases to Qwen names (`Portuguese`, `English`, `Auto`, …).

## Verification

- After edits: fully quit Electron so Python reloads this file.

## Child DOX Index

(none)
