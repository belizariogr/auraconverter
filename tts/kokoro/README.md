# TTS backend — Kokoro

Runtime leve com a mesma API HTTP do Qwen
(`/health`, `/tts`, `/tts/cancel`, `/tts/unload`).

| Plataforma | Backend | Qualidade |
|---|---|---|
| **macOS (Apple Silicon)** | **MLX** (`tts_server_mlx.py`) — `mlx-community/Kokoro-82M-bf16` | bf16 (completa) |
| Windows / Linux | ONNX Runtime (`tts_server.py`) + CUDA / MIGraphX / DirectML | ONNX fp |

No macOS o Kokoro **reutiliza o stack MLX do Qwen3** (`misaki[en]` + `mlx-audio`). O bundle do app **não inclui** `onnxruntime` / `kokoro_onnx`.

## Setup

```bash
# Windows / Linux — ONNX (+ GPU quando disponível)
bun run setup:tts:kokoro -- --force
bun run setup:tts:kokoro -- --force --accel=rocm   # AMD
bun run setup:tts:kokoro -- --force --accel=cuda   # NVIDIA
bun run setup:tts:kokoro -- --force --accel=cpu

# macOS — use o runtime MLX (já usado pelo Qwen3)
cd mlx && .venv/bin/pip install -r requirements.txt
```

## Modelos

Baixados pela UI:

- **macOS:** Hugging Face `mlx-community/Kokoro-82M-bf16` → `models/kokoro/Kokoro-82M-bf16/`
- **Win/Linux:** `kokoro-v1.0.onnx` + `voices-v1.0.bin` (release kokoro-onnx v1.0)

## Smoke

```bash
# macOS (MLX)
mlx/.venv/bin/python tts/kokoro/tts_server_mlx.py
curl -s http://127.0.0.1:8765/health | jq .backend,.device

# Windows / Linux (ONNX)
bun run start:tts:kokoro
curl -s http://127.0.0.1:8765/health | jq .device,.onnxProviders
```

## Warm-up GPU (AMD MIGraphX — só ONNX)

Na UI: **Aquecer GPU (pré-compilar)** — não se aplica ao backend MLX.
