## Purpose

Non-Qwen-MLX TTS backends: Kokoro (macOS MLX + cross-platform ONNX) and Qwen PyTorch (Windows/Linux).

## Ownership

Parent index only. Launch selection lives in root `server.ts` (`resolveKokoroLaunch` / `resolveQwenLaunch`).

## Local Contracts

- Same HTTP surface as Qwen MLX: `/health`, `/tts`, `/tts/cancel`, `/tts/unload`.
- `/tts` `language` comes from the book locale (see root `narrationLanguage.ts`); do not ignore it.
- Darwin Kokoro MLX shares `qwen3-tts-apple-silicon/.venv`.

## Work Guidance

- Engine id is `qwen3` | `kokoro` in `tts-engine.json` under `AURA_DATA_DIR`.

## Verification

(none beyond parent lint)

## Child DOX Index

- `.dox/tts/kokoro/AGENTS.md` — `tts/kokoro/`
- `.dox/tts/torch/AGENTS.md` — `tts/torch/`
