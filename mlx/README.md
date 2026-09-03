# Runtime MLX do Breeze TTS 2

Runtime local para macOS Apple Silicon usado pelo AuraReader. O modelo padrão é
`mlx-community/Breeze-TTS-2-mlx`, baixado pela tela de instalação do aplicativo
para `mlx/models/Breeze-TTS-2-mlx/`.

## Preparação

```bash
bun run setup:tts:mlx
```

O comando cria `mlx/.venv` com Python portátil e instala MLX, o tokenizer de
áudio e as dependências do servidor.

## Execução manual

```bash
cd mlx
.venv/bin/python tts_server.py
```

O servidor expõe `/health`, `/tts`, `/tts/cancel` e `/tts/unload` na porta
`8765`. O Express envia o prompt da voz para gerar a prévia; a prévia salva é
usada como áudio de referência para os blocos seguintes da narração.

O Breeze TTS 2 suporta inglês e chinês e é distribuído sob licença de pesquisa
e uso não comercial. Consulte `BREEZE_RUNTIME_LICENSE` e a licença baixada
junto com os pesos antes de redistribuir o aplicativo ou os resultados.
