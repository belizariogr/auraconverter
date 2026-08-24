## Purpose

Qwen3-TTS PyTorch FastAPI for Windows/Linux (`tts/torch/tts_server.py`).

## Ownership

CUDA / ROCm / CPU venvs via `scripts/setup-torch-tts.cjs`.

## Local Contracts

- Same JSON `/tts` body as MLX (including optional `topK` / `topP` / `repetitionPenalty` / `maxTokens`). CustomVoice path; ICL/ref audio unused unless Base is wired the same way as MLX.
- Sampling defaults match official Qwen3-TTS: `temperature=0.9`, `top_k=50`, `top_p=1.0`, `repetition_penalty=1.05`, `max_new_tokens=2048`, `do_sample=True`.
- Preview fallback paths match Node under `VOICE_PREVIEW_DIR` (runtime `AURA_DATA_DIR/assets/voice-previews`): `{voice}_{locale}_s100_{QWEN_TTS_PREVIEW_CACHE_VERSION}` (also accepts legacy without `sNNN`; default version `1.7b-narrate-v1`). Not bundled.
- Default weights: `Qwen3-TTS-12Hz-1.7B-Base` + `Qwen3-TTS-12Hz-1.7B-CustomVoice`.
- Resolve `/tts` `language` aliases (BCP-47 and names) via `LANGUAGE_MAP` before `generate_*`.
- Do **not** flatten `\n` in `synthesize()`; Node uses `\n\n\n` instead of `<break>` for Qwen.
- Not used on darwin (MLX Qwen wins).

## Work Guidance

- Keep speaker name mapping (`vivian` → `Vivian`) in sync with the MLX server list.

## Verification

(none in-repo)

## Child DOX Index

(none)
