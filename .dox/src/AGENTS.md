## Purpose

React UI (`vite`) for Aura Converter: document queue, voice/engine setup, narration progress, cover extract, MP3→M4B and M4B→MP3.

## Ownership

- `src/App.tsx` — narrate mode, document state, SSE to `/api/*`
- `src/ModePanels.tsx` — extract-cover / convert panels
- `src/ModelSetup.tsx` — engine, voice, model download
- `src/main.tsx`, `src/index.css` — bootstrap / theme

## Local Contracts

- Talk to the local Express server only (same origin in Electron).
- User-visible strings in pt-BR.
- Output format `mp3` | `m4b`; cover export is opt-in per document (`exportCover`).

## Work Guidance

- Keep PDF cover page and EPUB chapter selection in document state; do not invent a cover on the client.
- Audio download uses the server-saved file; M4B cover is inside the container, not a separate JPEG when format is m4b.

## Verification

- `bun run lint` (tsc). Exercise changed flows in the app when UI behavior changes.

## Child DOX Index

(none)
