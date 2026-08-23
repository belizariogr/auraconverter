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

export function repairExtractionHeuristics(text: string): string {
  if (!text) return "";
  return collapseSpacedLetters(joinHyphenatedLineBreaks(text)).replace(/[ ]{2,}/g, " ");
}

export function looksLikeBrokenExtraction(text: string): boolean {
  const tokens = text.split(/\s+/).filter(Boolean);
  if (tokens.length < 16) return false;
  const singles = tokens.filter((t) => t.length === 1 && /\p{L}/u.test(t)).length;
  return singles / tokens.length >= SINGLE_LETTER_RATIO;
}

function splitRepairChunks(text: string, maxChars = 1400): string[] {
  const paras = text.split(/\n+/);
  const chunks: string[] = [];
  let buf = "";
  for (const para of paras) {
    const next = buf ? `${buf}\n${para}` : para;
    if (next.length > maxChars && buf) {
      chunks.push(buf);
      buf = para;
    } else {
      buf = next;
    }
  }
  if (buf) chunks.push(buf);
  return chunks;
}

function acceptableModelOutput(original: string, repaired: string): boolean {
  const a = original.trim();
  const b = repaired.trim();
  if (!b) return false;
  if (b.length < a.length * 0.45) return false;
  if (b.length > a.length * 1.6) return false;
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
  if (!text || !looksLikeBrokenExtraction(text)) return text;

  const script = findRepairScript(options.auraRoot);
  if (!script || !options.python) {
    console.warn("[TextRepair] Broken glyphs remain, but no small model runtime is available.");
    return text;
  }

  const chunks = splitRepairChunks(text);
  const toFix = chunks.map((c, i) => (looksLikeBrokenExtraction(c) ? i : -1)).filter((i) => i >= 0);
  if (toFix.length === 0) return text;

  console.log(
    `[TextRepair] Running Qwen2.5-0.5B on ${toFix.length}/${chunks.length} chunk(s) with spaced letters.`
  );

  const payload = JSON.stringify({ chunks: toFix.map((i) => chunks[i]) });
  const cacheDir = path.join(options.dataDir, "models", "text-repair");

  try {
    const raw = await runRepairPython(options.python, script, payload, cacheDir);
    const parsed = JSON.parse(raw) as { chunks?: string[] };
    const fixed = parsed.chunks;
    if (!Array.isArray(fixed) || fixed.length !== toFix.length) return text;

    const out = chunks.slice();
    for (let n = 0; n < toFix.length; n++) {
      const idx = toFix[n];
      const candidate = String(fixed[n] ?? "").trim();
      if (acceptableModelOutput(chunks[idx], candidate)) {
        out[idx] = candidate;
      }
    }
    return out.join("\n");
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
    if (python.home) env.PYTHONHOME = python.home;
    if (python.pythonPath) env.PYTHONPATH = python.pythonPath;

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
