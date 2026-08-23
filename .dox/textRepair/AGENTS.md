## Purpose

Optional local LLM pass over extracted PDF/EPUB text (`textRepair/repair.py`, called from `textRepair.ts`).

## Ownership

Python helper only. Enable/disable and chunking stay in root `textRepair.ts`.

## Local Contracts

- Must not invent plot; repair OCR/layout only.
- Failures fall back to unrepaired extracted text.
- Model output must be book text only. Strip assistant preambles such as "Aqui está a correção do texto:" (`stripRepairPreamble` / `_strip_fences`).
- Lines of only digits and spaces (`42`, `1 0`, `1 3`), optional `()`/`[]`, are page numbers: strip at page edges (`stripPageEdgePagination`) and insert `\n\n\n` (`PAGE_NUMBER_GAP`). The OCR-repair model must keep those gaps (`split(/(\\n{2,})/)`).

## Work Guidance

(none extra)

## Verification

(none)

## Child DOX Index

(none)
