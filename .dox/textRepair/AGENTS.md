## Purpose

Optional local LLM pass over extracted PDF/EPUB text (`textRepair/repair.py`, called from `textRepair.ts`).

## Ownership

Python helper only. Enable/disable and chunking stay in root `textRepair.ts`.

## Local Contracts

- Must not invent plot; repair OCR/layout only.
- Failures fall back to unrepaired extracted text.

## Work Guidance

(none extra)

## Verification

(none)

## Child DOX Index

(none)
