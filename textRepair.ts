/**
 * Repair PDF/EPUB extraction artifacts (tracked letter-spacing, hyphenation).
 * Heuristics run always; a small open instruct model (Qwen 0.5B) only runs
 * when the text still looks like broken glyphs.
 */
import { spawn } from "child_process";
import fs from "fs";
import path from "path";

export type TextRepairPython = { bin: string; home?: string; pythonPath?: string };

const SINGLE_LETTER_RATIO = 0.12;
const MODEL_TIMEOUT_MS = 180_000;

/** Join runs of 4+ single-letter tokens: "p a l a v r a" → "palavra". */
export function collapseSpacedLetters(text: string): string {
  return text.replace(
    /(?<!\S)(?:[\p{L}\p{N}]\s+){3,}[\p{L}\p{N}][.,;:!?…]*(?!\S)/gu,
    (m) => m.replace(/\s+/g, "")
  );
}

export function joinHyphenatedLineBreaks(text: string): string {
  return text.replace(/(\p{L})-\s*\n\s*(\p{L})/gu, "$1$2");
}

/**
 * Pagination footer/header: digits and spaces only, optional wrappers
 * like `)`, `(10)`, `1 0`. Letters keep the line as real copy.
 */
export function isStandalonePageNumberLine(line: string): boolean {
  const trimmed = line.trim();

  if (!trimmed) {
    return false;
  }

  if (!/\d/.test(trimmed)) {
    return false;
  }

  const unwrapped = trimmed
    .replace(/^[\s(\[{«"'“''.\-–—•·]+/u, "")
    .replace(/[\s)\]}»"'”''.\-–—•·]+$/u, "")
    .trim();

  if (!unwrapped) {
    return false;
  }

  return /^[\d\s]+$/.test(unwrapped);
}

/**
 * PDF page numbers in this corpus sit on the first line (`9`, `1 0`).
 * Drop leading (and short trailing) pagination so they never join the body.
 */
export function stripPageEdgePagination(pageText: string): {
  text: string;
  strippedLeading: boolean;
} {
  if (!pageText) {
    return { text: "", strippedLeading: false };
  }

  const lines = pageText.replace(/\r\n/g, "\n").split("\n");
  let strippedLeading = false;

  while (lines.length > 0 && isStandalonePageNumberLine(lines[0] ?? "")) {
    lines.shift();
    strippedLeading = true;
  }

  while (lines.length > 0) {
    const last = lines[lines.length - 1] ?? "";

    if (!isStandalonePageNumberLine(last)) {
      break;
    }

    const digitCount = last.replace(/\D/g, "").length;

    if (digitCount < 1 || digitCount > 3) {
      break;
    }

    lines.pop();
  }

  return { text: lines.join("\n").trim(), strippedLeading };
}

/** Three newlines where a page number was removed (`\n\n\n`). */
export const PAGE_NUMBER_GAP = "\n\n\n";

/**
 * Drop the page-number line from narration and leave `\n\n\n` in the hole
 * so the extracted text has one extra blank line vs a normal paragraph.
 */
export function replaceStandaloneNumericLines(text: string): string {
  if (!text) {
    return "";
  }

  const lines = text.replace(/\r\n/g, "\n").split("\n");
  const out: string[] = [];

  for (const line of lines) {
    if (isStandalonePageNumberLine(line)) {
      if (out.length > 0 && out[out.length - 1] !== "") {
        out.push("");
      }

      out.push("");
      continue;
    }

    out.push(line);
  }

  return out.join("\n");
}

export function repairExtractionHeuristics(text: string): string {
  if (!text) {
    return "";
  }

  return replaceStandaloneNumericLines(
    collapseSpacedLetters(joinHyphenatedLineBreaks(text)).replace(/[ ]{2,}/g, " ")
  );
}

export function looksLikeBrokenExtraction(text: string): boolean {
  const tokens = text.split(/\s+/).filter(Boolean);

  if (tokens.length < 16) {
    return false;
  }

  const singles = tokens.filter((t) => t.length === 1 && /\p{L}/u.test(t)).length;
  return singles / tokens.length >= SINGLE_LETTER_RATIO;
}

const REPAIR_PREAMBLE_RE =
  /^(?:aqui est[áa]\s+(?:a\s+corre[cç][aã]o(?:\s+do\s+texto)?|o\s+texto\s+corrigido)|here(?:'|’| i)?s\s+the\s+corrected\s+text|segue\s+(?:a\s+corre[cç][aã]o|o\s+texto\s+corrigido)|texto\s+corrigido|corre[cç][aã]o(?:\s+do\s+texto)?)\s*:?\s*/i;

/** Drop assistant intros the 0.5B model sometimes prepends. */
export function stripRepairPreamble(text: string): string {
  let next = text.trim();

  for (let i = 0; i < 3; i++) {
    const stripped = next.replace(REPAIR_PREAMBLE_RE, "").replace(/^[\s:.\-—–]+/, "").trim();

    if (stripped === next) {
      break;
    }

    next = stripped;
  }

  return next;
}

function acceptableModelOutput(original: string, repaired: string): boolean {
  const a = original.trim();
  const b = repaired.trim();

  if (!b) {
    return false;
  }

  if (/^aqui est[áa]\b/i.test(b) && /corre[cç][aã]o|corrigido/i.test(b.slice(0, 80))) {
    return false;
  }

  if (b.length < a.length * 0.45) {
    return false;
  }

  if (b.length > a.length * 1.6) {
    return false;
  }

  return true;
}

function findRepairScript(auraRoot: string): string | null {
  const candidates = [
    path.join(auraRoot, "textRepair", "repair.py"),
    path.join(process.cwd(), "textRepair", "repair.py"),
  ];
  return candidates.find((p) => fs.existsSync(p)) || null;
}

export async function repairExtractedTextWithModel(
  text: string,
  options: {
    auraRoot: string;
    dataDir: string;
    python: TextRepairPython | null;
  }
): Promise<string> {
  if (!text || !looksLikeBrokenExtraction(text)) {
    return text;
  }

  const script = findRepairScript(options.auraRoot);

  if (!script || !options.python) {
    console.warn("[TextRepair] Broken glyphs remain, but no small model runtime is available.");
    return text;
  }

  const pieces = text.split(/(\n{2,})/);
  const textPieceIndexes: number[] = [];
  const toFix: string[] = [];

  for (let i = 0; i < pieces.length; i++) {
    const piece = pieces[i];

    if (!piece || /^\n+$/.test(piece) || !looksLikeBrokenExtraction(piece)) {
      continue;
    }

    textPieceIndexes.push(i);
    toFix.push(piece);
  }

  if (toFix.length === 0) {
    return text;
  }

  console.log(
    `[TextRepair] Running Qwen2.5-0.5B on ${toFix.length} paragraph(s) with spaced letters.`
  );

  const payload = JSON.stringify({ chunks: toFix });
  const cacheDir = path.join(options.dataDir, "models", "text-repair");

  try {
    const raw = await runRepairPython(options.python, script, payload, cacheDir);
    const parsed = JSON.parse(raw) as { chunks?: string[] };
    const fixed = parsed.chunks;

    if (!Array.isArray(fixed) || fixed.length !== toFix.length) {
      return text;
    }

    const out = pieces.slice();

    for (let n = 0; n < toFix.length; n++) {
      const idx = textPieceIndexes[n];
      const candidate = stripRepairPreamble(String(fixed[n] ?? ""));

      if (acceptableModelOutput(toFix[n], candidate)) {
        out[idx] = candidate;
      }

    }

    return out.join("");
  } catch (err: any) {
    console.warn("[TextRepair] Small model skipped:", err?.message || err);
    return text;
  }
}

function runRepairPython(
  python: TextRepairPython,
  script: string,
  payload: string,
  cacheDir: string
): Promise<string> {
  return new Promise((resolve, reject) => {
    const env: NodeJS.ProcessEnv = {
      ...process.env,
      TEXT_REPAIR_CACHE: cacheDir,
      HF_HOME: cacheDir,
      PYTHONNOUSERSITE: "1",
    };
    if (python.home) {
      env.PYTHONHOME = python.home;
    }

    if (python.pythonPath) {
      env.PYTHONPATH = python.pythonPath;
    }

    const child = spawn(python.bin, [script], {
      env,
      stdio: ["pipe", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";
    const timer = setTimeout(() => {
      child.kill("SIGTERM");
      reject(new Error("text repair model timed out"));
    }, MODEL_TIMEOUT_MS);

    child.stdout.on("data", (d) => {
      stdout += d.toString("utf8");
    });
    child.stderr.on("data", (d) => {
      stderr += d.toString("utf8");
    });
    child.on("error", (err) => {
      clearTimeout(timer);
      reject(err);
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      if (code !== 0) {
        reject(new Error(stderr.trim() || `repair.py exited ${code}`));
        return;
      }

      const line = stdout.trim().split("\n").filter(Boolean).pop() || "";

      if (!line) {
        reject(new Error(stderr.trim() || "empty repair.py output"));
        return;
      }
      resolve(line);
    });

    child.stdin.write(payload, "utf8");
    child.stdin.end();
  });
}
