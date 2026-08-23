## Purpose

Qwen3-TTS PyTorch FastAPI for Windows/Linux (`tts/torch/tts_server.py`).

## Ownership

CUDA / ROCm / CPU venvs via `scripts/setup-torch-tts.cjs`.

## Local Contracts

- Same JSON `/tts` body as MLX. CustomVoice path; ICL/ref audio unused unless Base is wired the same way as MLX.
- Not used on darwin (MLX Qwen wins).

## Work Guidance

- Keep speaker name mapping (`vivian` → `Vivian`) in sync with the MLX server list.

## Verification

(none in-repo)

## Child DOX Index

(none)
