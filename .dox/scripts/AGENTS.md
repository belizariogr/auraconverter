## Purpose

Build, pack, and TTS environment setup.

## Ownership

- `prepare-app-resources.cjs` — ffmpeg, python, models into `build/app-resources`
- `dist.cjs` / electron-builder invocation from `package.json`
- `setup-*-tts.cjs` — venv + pip for each backend
- `start.ts` — dev: TTS then `server.ts`

## Local Contracts

- Packaged app reads `AURA_ROOT` from extraResources `aura/`.
- Do not assume system ffmpeg in production; use `ffmpegBin.ts` resolution.
- `setup-mlx-tts.cjs` installs `mlx-audio` from PyPI (`requirements.txt`); git is not required for that pin.

## Work Guidance

- `bun run lint` does not typecheck `.cjs`.

## Verification

- `bun run lint` after TS that scripts spawn; smoke `prepare:mac` only when packaging.

## Child DOX Index

(none)
