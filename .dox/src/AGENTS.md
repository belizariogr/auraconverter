## Purpose

React UI (`vite`) for Aura Converter: document queue, voice/engine setup, narration progress, cover extract, MP3→M4B and M4B→MP3.

## Ownership

- `src/App.tsx` — narrate mode, document state (incl. per-doc `narrationLanguage`), global voice/speed prefs, SSE to `/api/*`
- `src/NarrationSpeedSlider.tsx` — global narration speed control (0.75–1.5×)
- `src/ModePanels.tsx` — extract-cover / convert panels
- `src/ModelSetup.tsx` — engine, voice, model download
- `src/main.tsx`, `src/index.css` — bootstrap / theme

## Local Contracts

- Talk to the local Express server only (same origin in Electron).
- User-visible strings in pt-BR.
- Output format `mp3` | `m4b`; cover export is opt-in per document (`exportCover`).
- Persist global `voice` (per engine) and `narrationSpeed` in `localStorage`; load both on app open; apply the same values to every document (never store speed/voice on the document or IndexedDB session).
- For Breeze, persist a custom narrator prompt per voice in `localStorage`; the edited prompt must be sent both to voice preview generation and to narration. The generated preview is the narration reference audio.
- Voice: hydrate from `localStorage` after `/api/tts-engine` returns the active engine + catalog; persist **only** when the user selects a voice — never on mount/hydrate (avoids overwriting the saved id with the first catalog entry).
- Persist last-used `narrationLanguage` in `localStorage` for new documents; also keep `narrationLanguage` per document in IndexedDB session restore (language stays per-book).
- Voice preview POST/DELETE `/api/voice-preview` sends `{ voiceName, language, speed }`; disk key is voice+locale+speed under `AURA_DATA_DIR/assets/voice-previews/` (gitignored, not packaged). Qwen: TTS@1× then `atempo`, saved WAV is the ICL anchor. Kokoro: native `speed` on `/tts`. Speed is applied only there — encode must not atempo again.
- Changing speed invalidates the in-UI preview blob cache (new sample for that rate).
- With Qwen, extracted/editable text must not contain `<break>` tags (use `\n\n\n` in their place).
- DELETE `/api/voice-preview` removes the saved WAV+TXT for that voice+locale+speed; UI regenerates on the next play (↺ button deletes then plays).
- `ModelSetup` shows the absolute `modelsDir` path so the user can delete weights manually if needed.
- `NarrationSpeedSlider` — range control for the global narration speed.
- On SSE `error` during TTS with `completed`/`total`, persist `narrationProgress` so resume retries the failed block (server discards that block's PCM and does not skip ahead).

## Work Guidance

- Keep PDF cover page and EPUB chapter selection in document state; do not invent a cover on the client.
- Audio download uses the server-saved file; M4B cover is inside the container, not a separate JPEG when format is m4b.

## Verification

- `bun run lint` (tsc). Exercise changed flows in the app when UI behavior changes.

## Child DOX Index

(none)
