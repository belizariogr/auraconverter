# TTS backend — MLX (macOS Apple Silicon)

O runtime MLX permanece em [`mlx/`](../../mlx/) (`tts_server.py` + venv/`site-packages`).

O app Node seleciona esse caminho automaticamente em `process.platform === "darwin"`.

Para Windows/Linux (PyTorch / CUDA / ROCm / CPU), veja [`tts/torch/`](../torch/).
