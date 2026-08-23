## Purpose

Electron shell: window, preload bridge, spawn of the bundled Node server and TTS child.

## Ownership

- `electron/main.cjs` — BrowserWindow, `AURA_ROOT` / `AURA_DATA_DIR`, server process
- `electron/preload.cjs` — isolated bridge only

## Local Contracts

- Packaged resources live under `process.resourcesPath/aura` (see `extraResources` in `package.json`).
- TTS env for Python must be set by the Node server (`server.ts`) when spawning the child, not only in the Electron process.

## Work Guidance

- Full quit is required after Python TTS server edits; a UI reload keeps the old Python process.

## Verification

- Packaged path: `bun run electron` after `build`.

## Child DOX Index

(none)
