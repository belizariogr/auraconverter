## Purpose

Runtime MLX do Breeze TTS 2 para macOS Apple Silicon, incluindo o servidor HTTP local e o código de inferência vendorizado.

## Ownership

- `tts_server.py` — API `/health`, `/tts`, `/tts/cancel` e `/tts/unload` usada pelo Express.
- `breeze_tts_mlx/` — inferência MLX e decodificação de áudio do Breeze TTS 2.
- `models/` — pesos baixados em tempo de execução; não deve ser versionado.

## Local Contracts

- O modelo padrão é `mlx-community/Breeze-TTS-2-mlx` em bf16.
- O Breeze suporta inglês e chinês; vozes são criadas por instrução e a prévia gerada é usada como referência de voz na narração.
- O runtime deve permanecer compatível com Apple Silicon e com o stack MLX compartilhado pelo Kokoro.
- Pesos, prévias e caches gerados não entram no Git nem no pacote do aplicativo.

## Work Guidance

- Mantenha o formato PCM `s16le` mono e a resposta JSON compatível com `synthesizeWithTts` em `server.ts`.
- Falhas de carregamento ou síntese devem ser propagadas pela API; não silencie erros do modelo.
- Preserve os avisos de licença do runtime e do modelo ao empacotar.

## Verification

- `python3 -m py_compile tts_server.py breeze_tts_mlx/*.py`
- `bun run lint`

## Child DOX Index

(none)
