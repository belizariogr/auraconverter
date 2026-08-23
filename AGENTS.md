## 1. Clean Code and TypeScript (Mandatory)
* **Readability First:** Code must be written for humans to read. Avoid complex one-liners or obscure logic just to save lines of code.
* **Strict Typing:** The use of `any` is strictly forbidden. Use interfaces, `types`, and Generics to ensure type safety.
* **Descriptive Naming:** Variables, functions, and classes must reveal their intent immediately. Do not use obscure abbreviations (e.g., use `userProfile` instead of `usrPrf`).
* **Single Responsibility:** Functions should do one thing and do it well. If a function exceeds 20-30 lines, verify if it is not taking on too many responsibilities.

## 2. Error Handling and Resilience
* **Fail-Fast:** Validate parameters and return errors at the very beginning of the function, before processing business rules or spawning ffmpeg / TTS.
* **No Silent Failures:** Never create empty `try/catch` blocks or blocks that merely use `console.log(error)`. Errors must be properly handled, formatted, and propagated to the correct layer. Cover extraction may warn-and-skip; encoding and TTS failures must reach the client.

## 3. AI Agent Behavior Instructions
* **Read the Context:** Before creating a new utility function, check if it does not already exist in root `*.ts` helpers (`mediaConvert.ts`, `coverExtract.ts`, `ffmpegBin.ts`, `ttsEngine.ts`).
* **No Incomplete Code:** Deliver the complete solution or the complete logical steps. Do not use placeholder comments like `// ... rest of the code here`.
* **Maintain the Standard:** Analyze the surrounding files and mimic the project's naming, export style, and Portuguese user-facing strings.
* **Always run the command:** `bun run lint`
* **EVERY COMMIT** must be written in pt-BR. Dont add comments on the commit message about DOX unless the commit changes only DOX.
* **Never switch workspace.** Stay in this repository (`auraconverter`).
* **Do not delete** `chunk-cache` under Application Support unless the user explicitly asks.

## Code formatting

- Always add a blank line (`\n`) after every `if` block. In case you touch a file with an `if` that is not following this pattern, you should fix it.

Examples:

**Right:**

```typescript
if (true)
    return;

if (true) {
    return;
}

const id = test
    ? 1
    : 2;
```

**Wrong:**

```typescript
if (true) return;
```

```typescript
if (true) { return }
```

# DOX framework

- DOX is highly performant AGENTS.md hierarchy installed here
- Agent must follow DOX instructions across any edits

## .dox Storage

Child docs live under `.dox/`, mirroring the scope directory they govern. Source trees stay free of scattered AGENTS.md files.

| Scope (subtree root) | Doc path |
|---|---|
| Repository root | `AGENTS.md` |
| `src/` | `.dox/src/AGENTS.md` |
| `electron/` | `.dox/electron/AGENTS.md` |

Rules:

- Only the root rail stays at `AGENTS.md`. Every other doc goes in `.dox/<mirrored-path>/AGENTS.md`.
- `<mirrored-path>` is the scope folder relative to the repo root, without a leading slash.
- Never create `AGENTS.md` beside source files or inside code directories.
- When creating, moving, or deleting a child doc, update the matching path under `.dox/` and keep the mirror aligned with the scope folder.
- Child DOX Index entries use the `.dox/...` path and name the scope folder they cover.

Resolution: for a target at `src/App.tsx`, walk `src/` and read `.dox/src/AGENTS.md` when present. The nearest applicable doc is the deepest mirror on that walk.

## Core Contract

- AGENTS.md files are binding work contracts for their subtrees
- Work products, source materials, instructions, records, assets, and durable docs must stay understandable from the nearest applicable AGENTS.md plus every parent AGENTS.md above it

## Read Before Editing

1. Read the root `AGENTS.md`
2. Identify every file or folder you expect to touch, in case it doesnt exists, create it for every file you touch
3. Walk from the repository root to each target path
4. Along each route, read every mirrored doc at `.dox/<path>/AGENTS.md` for directories on that walk
5. If a parent doc lists a child doc whose scope contains the path, read that child at its `.dox/...` path and continue from there
6. Use the nearest applicable doc as the local contract and parent docs for repo-wide rules
7. If docs conflict, the closer doc controls local work details, but no child doc may weaken DOX

Do not rely on memory. Re-read the applicable DOX chain in the current session before editing.

## Update After Editing

Every meaningful change requires a DOX pass before the task is done.

Update the closest owning doc when a change affects:

- purpose, scope, ownership, or responsibilities
- durable structure, contracts, workflows, or operating rules
- required inputs, outputs, permissions, constraints, side effects, or artifacts
- user preferences about behavior, communication, process, organization, or quality
- doc creation, deletion, move, rename, or Child DOX Index contents under `.dox/`

Update parent docs when parent-level structure, ownership, workflow, or child index changes. Update child docs under `.dox/` when parent changes alter local rules. Remove stale or contradictory text immediately. Small edits that do not change behavior or contracts may leave docs unchanged, but the DOX pass still must happen.

If DOX doc doest not exists, you have to create it!!

## Hierarchy

- Root `AGENTS.md` is the DOX rail: project-wide instructions, global preferences, durable workflow rules, and the top-level Child DOX Index
- Child docs live at `.dox/<mirrored-path>/AGENTS.md` and own domain-specific instructions plus their own Child DOX Index
- Each parent explains what its direct children cover and what stays owned by the parent
- The closer a doc is to the work, the more specific and practical it must be

## Child Doc Shape

- Create a child doc at `.dox/<mirrored-path>/AGENTS.md` when a folder becomes a durable boundary with its own purpose, rules, responsibilities, workflow, materials, or quality standards
- Work Guidance must reflect the current standards of the project or user instructions; if there are no specific standards or instructions yet, leave it empty
- Verification must reflect an existing check; if no verification framework exists yet, leave it empty and update it when one exists

Default section order:
- Purpose
- Ownership
- Local Contracts
- Work Guidance
- Verification
- Child DOX Index

## Style

- Keep docs concise, current, and operational
- Document stable contracts, not diary entries
- Put broad rules in parent docs and concrete details in child docs
- Prefer direct bullets with explicit names
- Do not duplicate rules across many files unless each scope needs a local version
- Delete stale notes instead of explaining history
- Trim obvious statements, repeated rules, misplaced detail, and warnings for risks that no longer exist

## Closeout

1. Re-check changed paths against the DOX chain (resolve child docs via `.dox/` mirror)
2. Update nearest owning docs and any affected parents or children under `.dox/`
3. Refresh every affected Child DOX Index
4. Remove stale or contradictory text and delete orphaned `.dox/` mirrors when scopes are removed
5. Run existing verification when relevant
6. Report any docs intentionally left unchanged and why

## Purpose (this repo)

Aura Converter / AuraReader: Electron + Express app that extracts PDF/EPUB text, narrates locally (Qwen3-TTS MLX on Mac, or Kokoro), and writes **MP3** or **M4B**. Root TypeScript (`server.ts`, `mediaConvert.ts`, `coverExtract.ts`, TTS launch helpers) is owned here.

## User Preferences

When the user requests a durable behavior change, record it here or in the relevant child doc under `.dox/`

- UI copy and API errors in **pt-BR**; code identifiers in English
- M4B must embed **one** cover: the OPF `cover-image` / largest cover-like JPEG — never every EPUB illustration (Books/QuickTime treat extra JPEGs as video)
- Encode AAC/MP3 at native TTS rate when possible; do not upsample 24 kHz → 48 kHz with a harsh `aresample` cutoff
- Apple M5: mlx-audio `mx.clear_cache()` must not run mid-`generate()`. Pin mlx ≥ 0.32.1 (0.30.3 drops mid-utterance Qwen audio). Do **not** force `MLX_ENABLE_TF32=0`.
- Qwen TTS: never insert `<break>` tags in extracted or editable text — use `\n\n\n` instead. Kokoro still uses `<break>` as silent PCM.
- Chunk PCM cache under Application Support is kept after encode unless the user asks to delete it
- Do not commit secrets (`.env`, credentials)

## Child DOX Index

Parents first; leafs with durable contracts sit under each parent.

### App

- `.dox/src/AGENTS.md` — `src/` (React UI: narrate, cover extract, MP3↔M4B)
- `.dox/electron/AGENTS.md` — `electron/` (main/preload; spawn bundled Node server)

### TTS

- `.dox/qwen3-tts-apple-silicon/AGENTS.md` — `qwen3-tts-apple-silicon/` (Qwen MLX FastAPI)
- `.dox/tts/AGENTS.md` — `tts/` (Kokoro MLX/ONNX, Torch Qwen)
- `.dox/tts/kokoro/AGENTS.md` — `tts/kokoro/`
- `.dox/tts/torch/AGENTS.md` — `tts/torch/`

### Tooling

- `.dox/scripts/AGENTS.md` — `scripts/` (prepare resources, dist, TTS setup)
- `.dox/textRepair/AGENTS.md` — `textRepair/` (optional LLM text cleanup)
