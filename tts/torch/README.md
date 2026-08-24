# TTS backend — PyTorch (Windows / Linux)

Backend oficial [`qwen-tts`](https://github.com/QwenLM/Qwen3-TTS) com a mesma API HTTP do MLX (`/health`, `/tts`, `/tts/cancel`, `/tts/unload`).

## Aceleradores

| `--accel` | GPU | Índice / wheels |
|---|---|---|
| `cuda` (default) | NVIDIA | `https://download.pytorch.org/whl/cu124` |
| `rocm` | AMD (RDNA 3/4, AI PRO, Ryzen AI suportados) | Linux: PyTorch ROCm; Windows: wheels AMD `repo.radeon.com` (Python **3.12**) |
| `cpu` | — | `https://download.pytorch.org/whl/cpu` |

Driver/ROCm/Adrenalin fica com o usuário; o app não embute o driver.

## Dev (venv)

```bash
# Na raiz do repo — baixa Python 3.12 portátil e instala torch + qwen-tts
bun run setup:tts              # detecta GPU (AMD→rocm, NVIDIA→cuda, senão cpu)
bun run setup:tts:rocm         # forçar AMD
bun run setup:tts:cuda
bun run setup:tts:cpu

bun run electron:dev
# ou só o servidor TTS:
bun run start:tts:torch
```

Instalação manual (alternativa):

```bash
cd tts/torch
# precisa de Python 3.12 (não 3.14)
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-base.txt
# escolha um:
pip install -r requirements-cuda.txt --index-url https://download.pytorch.org/whl/cu124
# pip install -r requirements-rocm.txt --index-url https://download.pytorch.org/whl/rocm6.3
# pip install -r requirements-cpu.txt  --index-url https://download.pytorch.org/whl/cpu
```

Windows + AMD ROCm (exemplo ROCm 7.2):

```bat
pip install --no-cache-dir ^
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torch-2.9.1%%2Brocmsdk20260116-cp312-cp312-win_amd64.whl
```

Modelos oficiais (baixados pela UI do app ou manualmente):

- `Qwen/Qwen3-TTS-12Hz-1.7B-Base` → `models/Qwen3-TTS-12Hz-1.7B-Base`
- `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` → `models/Qwen3-TTS-12Hz-1.7B-CustomVoice`

```bash
# a partir da raiz do repo
bun run start:app
# ou só o servidor TTS:
tts/torch/.venv/bin/python tts/torch/tts_server.py
```

Dependência de sistema: `sox` (qwen-tts avisa se faltar). No Arch: `sudo pacman -S sox`.

## Empacotamento

Rodar **na plataforma alvo** (wheels nativos). O prepare baixa um CPython **3.12** portátil
(python-build-standalone), cria `tts/torch/.venv`, e monta `build/app-resources/` no mesmo
estilo do Mac (`Resources/aura`).

```bash
bun install
bun run dist                 # OS atual; accel=cuda por padrão
AURA_TTS_ACCEL=rocm bun run dist
AURA_TTS_ACCEL=cpu bun run dist

# ou explícito:
bun run dist:linux           # / dist:linux:rocm / dist:linux:cpu
bun run dist:win             # / dist:win:rocm / dist:win:cpu
```

Saída em `release/` (`linux-unpacked/`, AppImage, deb / NSIS).

## Device em runtime

Ordem: `AURA_TTS_DEVICE` (`cpu` / `cuda` / `hip`) → senão CUDA/HIP se disponível → senão CPU.

`/health` reporta `device`, `accel` (`cuda`|`rocm`|`cpu`) e `gpu`.
