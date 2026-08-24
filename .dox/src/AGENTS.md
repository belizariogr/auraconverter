## Purpose

React UI (`vite`) for Aura Converter: document queue, voice/engine setup, narration progress, cover extract, MP3→M4B and M4B→MP3.

## Ownership

- `src/App.tsx` — narrate mode, document state (incl. `narrationLanguage`, `narrationSpeed`), SSE to `/api/*`
- `src/NarrationSpeedSlider.tsx` — per-document narration speed control (0.75–1.5×)
- `src/ModePanels.tsx` — extract-cover / convert panels
- `src/ModelSetup.tsx` — engine, voice, model download
- `src/main.tsx`, `src/index.css` — bootstrap / theme

## Local Contracts

- Talk to the local Express server only (same origin in Electron).
- User-visible strings in pt-BR.
- Output format `mp3` | `m4b`; cover export is opt-in per document (`exportCover`).
- Persist last-used `voice` (per engine), `narrationLanguage`, and `narrationSpeed` in `localStorage`; new documents inherit those prefs (language falls back to OS locale only if never set).
- Also keep `narrationLanguage` / `narrationSpeed` per document in IndexedDB session restore.
- Voice preview POST/DELETE `/api/voice-preview` sends `{ voiceName, language, speed }`; disk key is voice+locale+speed (`…_s100_…`). Qwen: TTS@1× then `atempo`, saved WAV is the ICL anchor. Kokoro: native `speed` on `/tts`. Speed is applied only there — encode must not atempo again.
- Changing speed invalidates the in-UI preview blob cache (new sample for that rate).
- With Qwen, extracted/editable text must not contain `<break>` tags (use `\n\n\n` in their place).
- DELETE `/api/voice-preview` removes the saved WAV+TXT for that voice+locale+speed; UI regenerates on the next play (↺ button deletes then plays).
- `ModelSetup` shows the absolute `modelsDir` path so the user can delete weights manually if needed.
- `NarrationSpeedSlider` — range control for per-document narration speed.

## Work Guidance

- Keep PDF cover page and EPUB chapter selection in document state; do not invent a cover on the client.
- Audio download uses the server-saved file; M4B cover is inside the container, not a separate JPEG when format is m4b.

## Verification

- `bun run lint` (tsc). Exercise changed flows in the app when UI behavior changes.

## Child DOX Index

(none)
