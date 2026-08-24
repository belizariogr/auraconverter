import express from "express";
import path from "path";
import fs from "fs";
import os from "os";
import { createHash } from "crypto";
import { spawn, type ChildProcess } from "child_process";
import multer from "multer";
import dotenv from "dotenv";
import { PDFDocument } from "pdf-lib";
// @ts-ignore
import { EPub } from "epub2";
import {
  cancelModelDownload,
  deleteModels,
  downloadMissingModels,
  getModelsStatus as getModelsStatusFromManager,
  isModelDownloadActive,
  resolveKokoroWeightsDir,
  resolveModelsDir,
} from "./modelManager";
import { detectGpu } from "./gpuDetect";
import { kokoroGpuLibraryPath, probeKokoroAccel } from "./kokoroAccel";
import {
  isTtsEngineId,
  readTtsEngine,
  writeTtsEngine,
  readKokoroDevice,
  writeKokoroDevice,
  isKokoroDeviceId,
  isKokoroBackendId,
  readKokoroBackend,
  writeKokoroBackend,
  type TtsEngineId,
} from "./ttsEngine";
import {
  languagePayloadForTts,
  mergeQwenInstruct,
  previewTextForLanguage,
  resolveNarrationLanguageId,
  voicePreviewCacheTag,
  type NarrationLanguageId,
} from "./narrationLanguage";
import {
  convertImageToJpeg,
  coverToBase64Jpeg,
  extractCover,
  extractEpubImages,
  getEpubChapterPreview,
} from "./coverExtract";
import {
  ensureFfmpeg,
  m4bToMp3AndCover,
  mp3ToM4b,
  pcmToM4b,
  pcmToMp3,
} from "./mediaConvert";
import {
  isStandalonePageNumberLine,
  PAGE_NUMBER_GAP,
  repairExtractedTextWithModel,
  repairExtractionHeuristics,
  replaceStandaloneNumericLines,
  stripPageEdgePagination,
  type TextRepairPython,
} from "./textRepair";

function resolveBundledPython(auraRoot: string): { bin: string; home: string } | null {
  const envHome = process.env.AURA_PYTHON_HOME;
  const homes = [
    envHome,
    path.join(auraRoot, "python", "Python.framework", "Versions", "3.12"),
    path.join(auraRoot, "python"),
  ].filter((h): h is string => Boolean(h));

  const binNames = (home: string) => [
    path.join(home, "bin", "python3.12"),
    path.join(home, "bin", "python3"),
    path.join(home, "bin", "python"),
    path.join(home, "python.exe"),
  ];

  for (const home of homes) {
    const bin = binNames(home).find((p) => fs.existsSync(p));
    if (bin) return { bin, home };
  }
  return null;
}

/** Packaged app root (Resources/aura) or project cwd. */
const AURA_ROOT = process.env.AURA_ROOT
  ? path.resolve(process.env.AURA_ROOT)
  : process.cwd();
/** Writable data (cache, .env) — userData when packaged, else project root. */
const AURA_DATA_DIR = process.env.AURA_DATA_DIR
  ? path.resolve(process.env.AURA_DATA_DIR)
  : AURA_ROOT;

dotenv.config({ path: path.join(AURA_DATA_DIR, ".env") });
dotenv.config({ path: path.join(AURA_ROOT, ".env") });
dotenv.config();

const app = express();
const PORT = Number(process.env.PORT || 3000);
const TTS_URL = process.env.TTS_URL || process.env.DIA_URL || "http://127.0.0.1:8765";
const TTS_PORT =
  process.env.TTS_PORT ||
  process.env.DIA_PORT ||
  (() => {
    try {
      return new URL(TTS_URL).port || "8765";
    } catch {
      return "8765";
    }
  })();
/** Bun idle-timeout for /tts (ms). 0 / unset = disable (Qwen can take a while per chunk). */
const TTS_TIMEOUT_MS = Number(process.env.TTS_TIMEOUT_MS ?? process.env.DIA_TTS_TIMEOUT_MS ?? "0");
/** Stable narrator style applied to every chunk (must match Qwen server default intent). */
const TTS_INSTRUCT =
  process.env.QWEN_TTS_INSTRUCT ||
  "Speak as a consistent, calm, neutral book narrator. Keep the same pitch, energy, emotion, and pace for every sentence. Do not sound excited, dramatic, whispery, or casual.";
/** Qwen3-TTS official / mlx-audio default sampling temperature. Override with QWEN_TTS_TEMPERATURE. */
const TTS_TEMPERATURE = Number(process.env.QWEN_TTS_TEMPERATURE ?? "0.9");
const TTS_LANGUAGE = process.env.QWEN_TTS_LANGUAGE || "Auto";
/** Bump when preview text, instruct, or ICL scheme changes so old disk samples and chunk caches are ignored. */
const VOICE_PREVIEW_CACHE_VERSION =
  process.env.QWEN_TTS_PREVIEW_CACHE_VERSION || "1.7b-br-v1";

let ttsChild: ChildProcess | null = null;
let ttsStartPromise: Promise<void> | null = null;
/** Engine currently bound to ttsChild (if any). */
let ttsRunningEngine: TtsEngineId | null = null;

type TtsAccel = "cuda" | "rocm" | "cpu" | "mlx";

function readTtsAccel(): TtsAccel {
  const metaPath = path.join(AURA_ROOT, "tts-accel.json");
  try {
    if (fs.existsSync(metaPath)) {
      const raw = JSON.parse(fs.readFileSync(metaPath, "utf8")) as { accel?: string };
      const accel = String(raw.accel || "").toLowerCase();
      if (accel === "cuda" || accel === "rocm" || accel === "cpu" || accel === "mlx") {
        return accel;
      }
    }
  } catch {
    // ignore malformed meta
  }
  if (process.platform === "darwin") return "mlx";
  return "cuda";
}

function rocmSpawnEnv(): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = {
    MIOPEN_FIND_MODE: process.env.MIOPEN_FIND_MODE ?? "2",
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:
      process.env.TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL ?? "1",
  };

  // Consumer RDNA2 (RX 6000 / Navi 21–23) often needs an override on ROCm.
  if (!process.env.HSA_OVERRIDE_GFX_VERSION) {
    try {
      const gpu = detectGpu(AURA_ROOT);
      const blob = gpu.devices.map((d) => d.name).join(" ").toLowerCase();
      if (
        /navi 2[123]|rx 6[789]\d{2}|rx 6800|rx 6900|rx 6700|rx 6600|gfx1030|gfx1031|gfx1032/.test(
          blob
        )
      ) {
        env.HSA_OVERRIDE_GFX_VERSION = "10.3.0";
      }
    } catch {
      // ignore
    }
  }

  return env;
}

function activeTtsEngine(): TtsEngineId {
  return readTtsEngine(AURA_DATA_DIR);
}

function resolveNarrationLanguageIdFromRequest(raw: unknown): NarrationLanguageId {
  const requested = typeof raw === "string" ? raw.trim() : "";

  return resolveNarrationLanguageId(requested || TTS_LANGUAGE);
}

function resolveNarrationLanguagePayload(raw: unknown): string {
  return languagePayloadForTts(
    resolveNarrationLanguageIdFromRequest(raw),
    activeTtsEngine(),
    readKokoroBackend(AURA_DATA_DIR)
  );
}

function resolveKokoroLaunch(): {
  command: string;
  args: string[];
  cwd: string;
  env: NodeJS.ProcessEnv;
} {
  const ttsDir = path.join(AURA_ROOT, "tts", "kokoro");
  const modelsDir = resolveKokoroWeightsDir(AURA_ROOT, AURA_DATA_DIR);
  const kokoroDevice = readKokoroDevice(AURA_DATA_DIR);
  const kokoroBackend = readKokoroBackend(AURA_DATA_DIR);

  if (kokoroBackend === "mlx" && process.platform === "darwin") {
    const mlxDir = path.join(AURA_ROOT, "qwen3-tts-apple-silicon");
    const sitePackages = path.join(mlxDir, "site-packages");
    const scriptName = "tts_server_mlx.py";
    if (!fs.existsSync(path.join(ttsDir, scriptName))) {
      throw new Error(
        `Kokoro MLX server missing: ${path.join(ttsDir, scriptName)}`
      );
    }

    const baseEnv: NodeJS.ProcessEnv = {
      ...process.env,
      QWEN_TTS_PORT: String(TTS_PORT),
      TTS_PORT: String(TTS_PORT),
      QWEN_TTS_PRELOAD: process.env.QWEN_TTS_PRELOAD ?? "0",
      KOKORO_MODEL_DIR: modelsDir,
      KOKORO_DEVICE: kokoroDevice,
      AURA_KOKORO_BACKEND: "mlx",
    };

    const bundled = resolveBundledPython(AURA_ROOT);
    const venvPython = path.join(mlxDir, ".venv", "bin", "python");

    if (bundled && fs.existsSync(sitePackages)) {
      return {
        command: bundled.bin,
        args: [scriptName, "--host", "127.0.0.1", "--port", String(TTS_PORT)],
        cwd: ttsDir,
        env: {
          ...baseEnv,
          PYTHONHOME: bundled.home,
          PYTHONPATH: sitePackages,
          PYTHONNOUSERSITE: "1",
        },
      };
    }

    if (fs.existsSync(venvPython)) {
      return {
        command: venvPython,
        args: [scriptName, "--host", "127.0.0.1", "--port", String(TTS_PORT)],
        cwd: ttsDir,
        env: baseEnv,
      };
    }

    throw new Error(
      `Kokoro TTS (MLX) runtime não encontrado.\n` +
        `Precisa do stack MLX em ${mlxDir} (site-packages ou .venv).\n` +
        `O setup do Qwen3/MLX também prepara o Kokoro no macOS.`
    );
  }

  // ONNX Runtime (+ Core ML / CUDA / MIGraphX / DirectML / CPU).
  const sitePackages = path.join(ttsDir, "site-packages");
  const cacheRoot = path.join(AURA_DATA_DIR, "kokoro-ort-cache");
  const baseEnv: NodeJS.ProcessEnv = {
    ...process.env,
    ...(kokoroDevice === "gpu" ? rocmSpawnEnv() : {}),
    QWEN_TTS_PORT: String(TTS_PORT),
    TTS_PORT: String(TTS_PORT),
    QWEN_TTS_PRELOAD: process.env.QWEN_TTS_PRELOAD ?? "0",
    KOKORO_MODEL_DIR: modelsDir,
    KOKORO_DEVICE: kokoroDevice,
    AURA_KOKORO_BACKEND: "onnx",
    ORT_MIGRAPHX_CACHE_PATH: process.env.ORT_MIGRAPHX_CACHE_PATH || cacheRoot,
    ORT_MIGRAPHX_MODEL_CACHE_PATH:
      process.env.ORT_MIGRAPHX_MODEL_CACHE_PATH || cacheRoot,
  };
  if (kokoroDevice === "cpu") {
    baseEnv.AURA_ONNX_PROVIDER = "CPUExecutionProvider";
    baseEnv.ONNX_PROVIDER = "CPUExecutionProvider";
  } else {
    delete baseEnv.AURA_ONNX_PROVIDER;
    delete baseEnv.ONNX_PROVIDER;
    const libPath = kokoroGpuLibraryPath();
    if (libPath) baseEnv.LD_LIBRARY_PATH = libPath;
    if (!baseEnv.MIGRAPHX_GPU_COMPILE_PARALLEL) {
      baseEnv.MIGRAPHX_GPU_COMPILE_PARALLEL = String(Math.max(1, os.cpus().length));
    }
  }

  const pythonHome =
    process.env.AURA_PYTHON_HOME || path.join(AURA_ROOT, "python");
  const bundledCandidates = [
    path.join(pythonHome, "python.exe"),
    path.join(pythonHome, "bin", "python3.12"),
    path.join(pythonHome, "bin", "python3"),
    path.join(pythonHome, "bin", "python"),
    path.join(AURA_ROOT, "python", "Python.framework", "Versions", "3.12", "bin", "python3.12"),
  ];
  const bundledPython = bundledCandidates.find((p) => fs.existsSync(p));
  const venvCandidates = [
    path.join(ttsDir, ".venv", "Scripts", "python.exe"),
    path.join(ttsDir, ".venv", "bin", "python"),
  ];
  const venvPython = venvCandidates.find((p) => fs.existsSync(p));

  if (bundledPython && fs.existsSync(sitePackages)) {
    return {
      command: bundledPython,
      args: ["tts_server.py", "--host", "127.0.0.1", "--port", String(TTS_PORT)],
      cwd: ttsDir,
      env: {
        ...baseEnv,
        PYTHONHOME:
          bundledPython.includes("Python.framework")
            ? path.join(AURA_ROOT, "python", "Python.framework", "Versions", "3.12")
            : pythonHome,
        PYTHONPATH: sitePackages,
        PYTHONNOUSERSITE: "1",
      },
    };
  }

  if (venvPython) {
    return {
      command: venvPython,
      args: ["tts_server.py", "--host", "127.0.0.1", "--port", String(TTS_PORT)],
      cwd: ttsDir,
      env: baseEnv,
    };
  }

  throw new Error(
    `Kokoro TTS runtime não encontrado em ${ttsDir} (site-packages ou .venv).\n` +
      `Configure com: bun run setup:tts:kokoro`
  );
}

function resolveQwenLaunch(): {
  command: string;
  args: string[];
  cwd: string;
  env: NodeJS.ProcessEnv;
} {
  const previewDir = path.join(AURA_ROOT, "assets", "voice-previews");
  const modelsDir = resolveModelsDir(AURA_ROOT, AURA_DATA_DIR, "qwen3");
  const accel = readTtsAccel();
  const useTorch = process.platform === "win32" || process.platform === "linux";
  const ttsDir = useTorch
    ? path.join(AURA_ROOT, "tts", "torch")
    : path.join(AURA_ROOT, "qwen3-tts-apple-silicon");
  const sitePackages = path.join(ttsDir, "site-packages");

  const baseEnv: NodeJS.ProcessEnv = {
    ...process.env,
    QWEN_TTS_PORT: String(TTS_PORT),
    TTS_PORT: String(TTS_PORT),
    QWEN_TTS_PRELOAD: process.env.QWEN_TTS_PRELOAD ?? "0",
    QWEN_TTS_PREVIEW_CACHE_VERSION: VOICE_PREVIEW_CACHE_VERSION,
    VOICE_PREVIEW_DIR: previewDir,
    QWEN_TTS_MODELS_DIR: modelsDir,
    ...(accel === "rocm" ? rocmSpawnEnv() : {}),
  };

  if (useTorch) {
    const pythonHome =
      process.env.AURA_PYTHON_HOME || path.join(AURA_ROOT, "python");
    const bundledCandidates = [
      path.join(pythonHome, "python.exe"),
      path.join(pythonHome, "bin", "python3.12"),
      path.join(pythonHome, "bin", "python3"),
      path.join(pythonHome, "bin", "python"),
    ];
    const bundledPython = bundledCandidates.find((p) => fs.existsSync(p));
    const venvCandidates = [
      path.join(ttsDir, ".venv", "Scripts", "python.exe"),
      path.join(ttsDir, ".venv", "bin", "python"),
    ];
    const venvPython = venvCandidates.find((p) => fs.existsSync(p));

    if (bundledPython && fs.existsSync(sitePackages)) {
      return {
        command: bundledPython,
        args: ["tts_server.py", "--host", "127.0.0.1", "--port", String(TTS_PORT)],
        cwd: ttsDir,
        env: {
          ...baseEnv,
          PYTHONHOME: pythonHome,
          PYTHONPATH: sitePackages,
          PYTHONNOUSERSITE: "1",
        },
      };
    }

    if (venvPython) {
      return {
        command: venvPython,
        args: ["tts_server.py", "--host", "127.0.0.1", "--port", String(TTS_PORT)],
        cwd: ttsDir,
        env: baseEnv,
      };
    }

    throw new Error(
      `Qwen TTS (Torch) runtime não encontrado em ${ttsDir} (site-packages ou .venv).\n` +
        `Configure com: bun run setup:tts\n` +
        `(AMD → setup:tts -- --accel=rocm | NVIDIA → --accel=cuda | sem GPU → --accel=cpu). ` +
        `Detalhes: tts/torch/README.md (accel=${accel}).`
    );
  }

  // darwin / MLX — portable CPython (build/app-resources/python) or legacy framework.
  const bundled = resolveBundledPython(AURA_ROOT);
  const venvPython = path.join(ttsDir, ".venv", "bin", "python");

  if (bundled && fs.existsSync(sitePackages)) {
    return {
      command: bundled.bin,
      args: ["tts_server.py", "--host", "127.0.0.1", "--port", String(TTS_PORT)],
      cwd: ttsDir,
      env: {
        ...baseEnv,
        PYTHONHOME: bundled.home,
        PYTHONPATH: sitePackages,
        PYTHONNOUSERSITE: "1",
      },
    };
  }

  if (fs.existsSync(venvPython)) {
    return {
      command: venvPython,
      args: ["tts_server.py", "--host", "127.0.0.1", "--port", String(TTS_PORT)],
      cwd: ttsDir,
      env: baseEnv,
    };
  }

  throw new Error(
    `Qwen TTS runtime não encontrado em ${ttsDir} (site-packages ou .venv).`
  );
}

function resolveTtsLaunch(engine: TtsEngineId = activeTtsEngine()): {
  command: string;
  args: string[];
  cwd: string;
  env: NodeJS.ProcessEnv;
  engine: TtsEngineId;
} {
  if (engine === "kokoro") {
    return { ...resolveKokoroLaunch(), engine };
  }
  return { ...resolveQwenLaunch(), engine };
}

function getModelsStatus() {
  return getModelsStatusFromManager(AURA_ROOT, AURA_DATA_DIR);
}

async function isTtsProcessReady(engine?: TtsEngineId): Promise<boolean> {
  const want = engine ?? activeTtsEngine();
  if (ttsRunningEngine && ttsRunningEngine !== want) return false;
  try {
    const ttsRes = await fetch(`${TTS_URL}/health`);
    if (!ttsRes.ok) return false;
    const body = (await ttsRes.json()) as { ready?: boolean; provider?: string };
    if (body.ready === false) return false;
    if (want === "kokoro" && body.provider && body.provider !== "kokoro") return false;
    if (want === "qwen3" && body.provider === "kokoro") return false;
    return true;
  } catch {
    return false;
  }
}

/** Start the active TTS process on first need (preview / narrate). */
async function ensureTtsRunning(timeoutMs = 90_000): Promise<void> {
  const engine = activeTtsEngine();
  if (await isTtsProcessReady(engine)) return;
  if (ttsStartPromise) {
    await ttsStartPromise;
    return;
  }

  ttsStartPromise = (async () => {
    if (await isTtsProcessReady(engine)) return;

    // Wrong engine still listening — stop before respawn.
    if (ttsChild || (await isTtsProcessAlive())) {
      console.log(`[TTS] Restarting for engine=${engine} (was ${ttsRunningEngine || "unknown"})`);
      await unloadTtsModel().catch(() => undefined);
      stopManagedTts();
      await new Promise((r) => setTimeout(r, 400));
    }

    const status = getModelsStatus();
    if (!status.ready) {
      throw new Error(
        "Modelos TTS ainda não foram baixados. Abra a tela de instalação dos modelos."
      );
    }

    const launch = resolveTtsLaunch(engine);
    const label = engine === "kokoro" ? "Kokoro" : "Qwen3";

    if (!ttsChild || ttsChild.killed || ttsChild.exitCode !== null) {
      console.log(`[TTS] Starting ${label} TTS lazily on port ${TTS_PORT}...`);
      ttsRunningEngine = engine;
      ttsChild = spawn(launch.command, launch.args, {
        cwd: launch.cwd,
        env: launch.env,
        stdio: "inherit",
      });
      ttsChild.on("exit", (code, signal) => {
        console.warn(`[TTS] ${label} exited (code=${code}, signal=${signal})`);
        ttsChild = null;
        ttsRunningEngine = null;
      });
    }

    const started = Date.now();
    let lastLog = 0;
    while (Date.now() - started < timeoutMs) {
      if (await isTtsProcessReady(engine)) {
        console.log(`[TTS] ${label} TTS is ready (model loads on first conversion).`);
        return;
      }
      if (Date.now() - lastLog > 5_000) {
        console.log(`[TTS] still waiting for ${label} TTS server...`);
        lastLog = Date.now();
      }
      await new Promise((r) => setTimeout(r, 400));
    }
    throw new Error(`${label} TTS did not become ready within ${timeoutMs / 1000}s`);
  })().finally(() => {
    ttsStartPromise = null;
  });

  await ttsStartPromise;
}

async function isTtsProcessAlive(): Promise<boolean> {
  try {
    const ttsRes = await fetch(`${TTS_URL}/health`);
    return ttsRes.ok;
  } catch {
    return false;
  }
}

function stopManagedTts() {
  if (ttsChild && !ttsChild.killed) {
    try {
      ttsChild.kill("SIGTERM");
    } catch {
      // ignore
    }
  }
  ttsChild = null;
  ttsRunningEngine = null;
}

process.on("exit", stopManagedTts);
process.once("SIGINT", () => {
  stopManagedTts();
  process.exit(0);
});
process.once("SIGTERM", () => {
  stopManagedTts();
  process.exit(0);
});

// Set up larger limits for base64 file payloads (document extract / previews)
app.use(express.json({ limit: "200mb" }));
app.use(express.urlencoded({ limit: "200mb", extended: true }));

const convertUpload = multer({
  dest: path.join(os.tmpdir(), "aura-uploads"),
  limits: { fileSize: 2 * 1024 * 1024 * 1024 }, // 2 GB
});
fs.mkdirSync(path.join(os.tmpdir(), "aura-uploads"), { recursive: true });

/** Decode HTML named + numeric entities that often leak from PDF/EPUB extraction. */
function decodeHtmlEntities(raw: string): string {
  if (!raw) return "";

  const named: Record<string, string> = {
    nbsp: " ",
    lt: "<",
    gt: ">",
    amp: "&",
    quot: '"',
    apos: "'",
    ldquo: '"',
    rdquo: '"',
    lsquo: "'",
    rsquo: "'",
    ndash: "-",
    mdash: "—",
    hellip: "…",
    bull: "•",
  };

  let text = raw.replace(/&([a-zA-Z]+);/g, (match, name: string) => {
    const key = name.toLowerCase();
    return Object.prototype.hasOwnProperty.call(named, key) ? named[key] : match;
  });

  // Decimal: &#160; Hex: &#xA0; / &#XA0;
  text = text.replace(/&#(\d+);/g, (_, code: string) => {
    const n = Number(code);
    if (n === 160 || n === 0x202f || n === 0x2007) return " "; // nbsp-like
    try {
      return String.fromCodePoint(n);
    } catch {
      return "";
    }
  });

  text = text.replace(/&#x([0-9a-fA-F]+);/g, (_, hex: string) => {
    const n = parseInt(hex, 16);
    if (n === 0xa0 || n === 0x202f || n === 0x2007) return " ";
    try {
      return String.fromCodePoint(n);
    } catch {
      return "";
    }
  });

  return text;
}

/** Wrap a chapter/section title with em dashes for TTS pacing cues. */
function formatChapterTitle(title: string): string {
  const t = title.replace(/\s+/g, " ").trim();
  if (!t) return "";
  if (/^—\s*.+\s*—$/.test(t)) return t;
  return `— ${t} —`;
}

// Helper: Strip HTML tags and decode basic entities to get plain text from EPUB XHTML
function stripHtml(html: string): string {
  if (!html) return "";
  
  // Replace linebreaks and paragraphs; wrap headings with em dashes
  let text = html
    .replace(/<br\s*\/?>/gi, "\n")
    .replace(/<\/p>/gi, "\n\n")
    .replace(/<p[^>]*>/gi, "")
    .replace(/<div[^>]*>/gi, "")
    .replace(/<\/div>/gi, "\n")
    .replace(/<h[1-6][^>]*>([\s\S]*?)<\/h[1-6]>/gi, (_, inner: string) => {
      const title = decodeHtmlEntities(String(inner).replace(/<[^>]*>/g, " "))
        .replace(/\s+/g, " ")
        .trim();
      const wrapped = formatChapterTitle(title);
      return wrapped ? `\n\n${wrapped}\n\n` : "\n\n";
    });
    
  // Strip any other tags
  text = text.replace(/<[^>]*>/g, " ");
  text = decodeHtmlEntities(text);

  // Trim each line but keep blank lines so paragraph breaks survive into sanitize
  text = text
    .split("\n")
    .map((line) => line.trim())
    .join("\n");

  return sanitizeExtractedText(text);
}

const BREAK_LINE_RE = /^<break\s+time="([\d.]+)s"\s*\/?\s*>$/i;
const QWEN_BREAK_GAP = "\n\n\n";

function formatBreak(seconds: number): string {
  const clamped = Math.min(10, Math.max(0.25, seconds));
  const label = Number.isInteger(clamped) ? String(clamped) : clamped.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
  return `<break time="${label}s" />`;
}

/** Qwen has no SSML; `<break>` becomes three newlines. */
function replaceBreakTagsWithNewlines(text: string): string {
  return text
    .replace(/<break\s+time="[\d.]+s"\s*\/?\s*>/gi, QWEN_BREAK_GAP)
    .replace(/[ ]{2,}/g, " ")
    .replace(/\n{4,}/g, PAGE_NUMBER_GAP);
}

function textForEngine(
  text: string,
  engine: TtsEngineId = activeTtsEngine()
): string {
  if (engine !== "qwen3") {
    return text;
  }

  return replaceBreakTagsWithNewlines(text);
}

function pushQwenBreakGap(collapsed: string[]): void {
  if (collapsed.length === 0) {
    return;
  }

  const last = collapsed[collapsed.length - 1];
  const prev = collapsed[collapsed.length - 2];

  if (last === "" && prev === "") {
    return;
  }

  collapsed.push("", "");
}

/** Each empty line → 0.25s. Break markers are only emitted when this exceeds 1s. */
function pauseSecondsForEmptyLines(emptyLineCount: number): number {
  if (emptyLineCount < 1) return 0;
  return emptyLineCount * 0.25;
}

function isPauseLine(line: string): boolean {
  const t = line.trim();
  return t === "*" || t === "..." || BREAK_LINE_RE.test(t);
}

function parseBreakSeconds(line: string): number | null {
  const m = line.trim().match(BREAK_LINE_RE);
  if (!m) return null;
  const sec = Number(m[1]);
  return Number.isFinite(sec) && sec > 0 ? Math.min(10, sec) : 0.25;
}

function pauseSecondsFromMarker(line: string): number {
  return parseBreakSeconds(line) ?? 0.25;
}

/** Silent PCM s16le for `<break>` markers (TTS engines do not honor SSML). */
function silencePcmS16le(sampleRate: number, seconds: number): Buffer {
  const samples = Math.max(0, Math.floor(sampleRate * seconds));
  return Buffer.alloc(samples * 2);
}

function pushOrMergePause(collapsed: string[], mark: string): void {
  if (collapsed.length === 0) return; // drop leading pauses
  const last = collapsed[collapsed.length - 1];
  if (isPauseLine(last)) {
    const merged = Math.min(
      10,
      Math.max(pauseSecondsFromMarker(last), pauseSecondsFromMarker(mark))
    );
    collapsed[collapsed.length - 1] = formatBreak(merged);
    return;
  }
  collapsed.push(mark);
}

/** Remove control chars, markup leftovers, and OCR/LLM junk before TTS and display. */
function sanitizeExtractedText(
  raw: string,
  engine: TtsEngineId = activeTtsEngine()
): string {
  if (!raw) {
    return "";
  }

  const skipBreakTags = engine === "qwen3";

  let text = decodeHtmlEntities(raw)
    // Strip Markdown code fences and heading markers often added by LLMs
    .replace(/```[\s\S]*?```/g, (block) =>
      block.replace(/^```[a-zA-Z0-9]*\n?/, "").replace(/\n?```$/, "")
    )
    .replace(/^#{1,6}\s+/gm, "")
    .replace(/\*\*(.+?)\*\*/g, "$1")
    .replace(/(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)/g, "$1")
    .replace(/`([^`]+)`/g, "$1")
    // Common PDF extraction artifacts (page labels, etc.)
    .replace(/^\s*\[?\s*Page\s+\d+\s*\]?\s*:?\s*/gim, "")
    .replace(/^\s*P[aá]gina\s+\d+\s*:?\s*/gim, "")
    .replace(/\uFFFD/g, "") // replacement char
    .replace(/[\u200B-\u200D\uFEFF\u00AD]/g, "") // zero-width / soft hyphen
    .replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/g, "") // control chars (keep \t \n)
    .replace(/\t+/g, " ")
    // Collapse odd runs of punctuation / separators from OCR
    .replace(/[|¦]{2,}/g, " ")
    .replace(/_{3,}/g, " ")
    .replace(/-{3,}/g, "—")
    .replace(/[•●▪◦]+/g, "•")
    // Normalize whitespace (incl. real nbsp U+00A0)
    .replace(/[\u00A0\u202F\u2007]/g, " ")
    .replace(/[ ]{2,}/g, " ");

  const lines = text.split("\n").map((line) => line.trim());
  const collapsed: string[] = [];
  let emptyRun = 0;

  const flushEmptyRun = () => {
    if (emptyRun < 1) {
      return;
    }

    const count = emptyRun;
    const seconds = pauseSecondsForEmptyLines(count);
    emptyRun = 0;
    // Only inject silence when the gap is longer than 1s; otherwise keep plain \n.
    // Qwen: never emit `<break>` — insert `\n\n\n` instead.
    if (seconds > 1) {
      if (skipBreakTags) {
        pushQwenBreakGap(collapsed);
      } else {
        pushOrMergePause(collapsed, formatBreak(seconds));
      }

      return;
    }

    for (let i = 0; i < count; i++) {
      collapsed.push("");
    }
  };

  for (const line of lines) {
    if (!line) {
      emptyRun += 1;
      continue;
    }

    if (isStandalonePageNumberLine(line)) {
      emptyRun += 2;
      continue;
    }

    flushEmptyRun();

    if (isPauseLine(line)) {
      if (skipBreakTags) {
        pushQwenBreakGap(collapsed);
      } else {
        pushOrMergePause(collapsed, formatBreak(pauseSecondsFromMarker(line)));
      }

      continue;
    }

    collapsed.push(line);
  }
  // Trailing blank runs are dropped (no flush)

  while (collapsed.length > 0 && isPauseLine(collapsed[collapsed.length - 1])) {
    collapsed.pop();
  }

  const cleaned = repairExtractionHeuristics(collapsed.join("\n").trim());

  if (skipBreakTags) {
    return replaceBreakTagsWithNewlines(cleaned);
  }

  return cleaned;
}

// Helper: Extract Text from EPUB Chapters using local epub2 library
async function extractTextFromEpub(filePath: string, startChapter: number, endChapter: number): Promise<string> {
  return new Promise((resolve, reject) => {
    const epub = new EPub(filePath);
    
    epub.on("end", async () => {
      try {
        const flow = epub.flow;
        if (!flow || flow.length === 0) {
          reject(new Error("O arquivo EPUB não possui capítulos ou seções legíveis."));
          return;
        }

        const startIdx = Math.max(0, startChapter - 1);
        const endIdx = Math.min(flow.length - 1, endChapter - 1);

        if (startIdx >= flow.length) {
          reject(new Error(`O EPUB possui apenas ${flow.length} seções/capítulos, mas a seção inicial solicitada foi ${startChapter}.`));
          return;
        }
        if (endIdx < startIdx) {
          reject(new Error("Intervalo de seções/capítulos inválido."));
          return;
        }

        let fullText = "";
        
        for (let i = startIdx; i <= endIdx; i++) {
          const chapter = flow[i];
          const text = await new Promise<string>((resChan) => {
            epub.getChapter(chapter.id, (err, text) => {
              if (err) {
                console.warn(`[Epub] Error reading chapter ${chapter.id}:`, err);
                resChan("");
              } else {
                resChan(text || "");
              }
            });
          });

          if (text) {
            const chapterText = stripHtml(text);
            if (chapterText) {
              const rawTitle = String(chapter.title || "").trim();
              const wrappedTitle = formatChapterTitle(rawTitle);
              const firstLine = chapterText.split("\n")[0]?.trim() || "";
              // Prepend spine title when the chapter body doesn't already open with it
              const block =
                wrappedTitle &&
                firstLine !== wrappedTitle &&
                firstLine !== rawTitle
                  ? `${wrappedTitle}\n\n${chapterText}`
                  : chapterText;
              fullText += (fullText ? "\n\n" : "") + block;
            }
          }
        }

        resolve(sanitizeExtractedText(fullText));
      } catch (err) {
        reject(err);
      }
    });

    epub.on("error", (err) => {
      reject(err);
    });

    epub.parse();
  });
}

type EpubChapterInfo = { index: number; id: string; title: string };

async function getEpubOutline(filePath: string): Promise<EpubChapterInfo[]> {
  return new Promise((resolve, reject) => {
    const epub = new EPub(filePath);
    epub.on("end", () => {
      try {
        const flow = epub.flow || [];
        const chapters = flow.map((chapter: { id?: string; title?: string }, i: number) => ({
          index: i + 1,
          id: chapter.id || String(i),
          title: (chapter.title || "").trim() || `Seção ${i + 1}`,
        }));
        resolve(chapters);
      } catch (err) {
        reject(err);
      }
    });
    epub.on("error", reject);
    epub.parse();
  });
}

/**
 * Synthesize speech via the local Qwen3-TTS HTTP server.
 * Returns PCM s16le samples and sample rate.
 *
 * Pass refAudioPath + refText to lock speaker identity via ICL (Base model).
 * Pass skipIcl when generating the enrollment/preview sample itself.
 *
 * Note: Bun's fetch has a ~5min idle timeout by default. Qwen sends no bytes until
 * generation finishes, so we must disable/extend that timeout or the client
 * aborts while TTS keeps running.
 */
async function synthesizeWithTts(
  text: string,
  voice: string,
  jobId?: string,
  signal?: AbortSignal,
  opts?: {
    refAudioPath?: string;
    refText?: string;
    skipIcl?: boolean;
    language?: string;
    instruct?: string;
  }
): Promise<{ pcm: Buffer; sampleRate: number; cancelled: boolean; icl?: boolean }> {
  if (signal?.aborted) {
    return { pcm: Buffer.alloc(0), sampleRate: 24000, cancelled: true };
  }

  await ensureTtsRunning();

  const fetchOptions: RequestInit & { timeout?: boolean | number | { connect?: number | false; idle?: number | false } } = {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text,
      voice,
      jobId,
      instruct: (opts?.instruct || "").trim() || TTS_INSTRUCT,
      temperature: TTS_TEMPERATURE,
      language: opts?.language?.trim() || TTS_LANGUAGE,
      ...(opts?.refAudioPath && opts?.refText
        ? { refAudioPath: opts.refAudioPath, refText: opts.refText }
        : {}),
      ...(opts?.skipIcl ? { skipIcl: true } : {}),
    }),
    signal,
    timeout: TTS_TIMEOUT_MS > 0
      ? { connect: 30_000, idle: TTS_TIMEOUT_MS }
      : false,
  };

  let res: Response;
  try {
    res = await fetch(`${TTS_URL}/tts`, fetchOptions);
  } catch (err: any) {
    if (signal?.aborted || err?.name === "AbortError") {
      return { pcm: Buffer.alloc(0), sampleRate: 24000, cancelled: true };
    }
    if (jobId) {
      await cancelTtsJob(jobId);
    }
    const msg = err?.message || String(err);
    throw new Error(msg.includes("timed out") || err?.name === "TimeoutError"
      ? `Qwen TTS timed out waiting for generation (${msg}). Increase TTS_TIMEOUT_MS or leave it unset to disable.`
      : msg);
  }

  if (!res.ok) {
    let detail = `Qwen TTS HTTP ${res.status}`;
    try {
      const errBody = await res.json();
      detail = errBody.detail || errBody.error || detail;
    } catch {
      // ignore parse errors
    }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }

  const body = (await res.json()) as {
    sampleRate: number;
    audioData: string;
    format: string;
    cancelled?: boolean;
    icl?: boolean;
    voice?: string;
  };

  const cancelled = !!body.cancelled;
  const sampleRate = Number(body.sampleRate || 24000);
  if (body.format && body.format !== "pcm_s16le") {
    throw new Error(`Formato de áudio TTS inesperado: ${body.format}.`);
  }
  if (!Number.isInteger(sampleRate) || sampleRate < 8000 || sampleRate > 192000) {
    throw new Error(`Sample rate TTS inválido: ${body.sampleRate}.`);
  }
  if (body.voice && body.voice.toLowerCase() !== voice.toLowerCase().replace(/\s+/g, "_")) {
    console.warn(
      `[TTS] Voice mismatch: requested=${voice} resolved=${body.voice}`
    );
  }
  if (!body.audioData) {
    return {
      pcm: Buffer.alloc(0),
      sampleRate,
      cancelled,
      icl: !!body.icl,
    };
  }

  const pcm = Buffer.from(body.audioData, "base64");
  if (pcm.length % 2 !== 0) {
    throw new Error(`PCM16 TTS inválido: ${pcm.length} bytes.`);
  }
  return {
    pcm,
    sampleRate,
    cancelled,
    icl: !!body.icl,
  };
}

async function cancelTtsJob(jobId: string): Promise<void> {
  try {
    const res = await fetch(`${TTS_URL}/tts/cancel`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ jobId }),
    });
    if (!res.ok) {
      console.warn(`[NarrateStop] TTS cancel HTTP ${res.status} for job ${jobId}`);
      return;
    }
    const body = (await res.json()) as { success?: boolean; message?: string };
    console.log(`[NarrateStop] TTS cancel response:`, body);
  } catch (err) {
    console.warn(`[NarrateStop] Failed to cancel TTS job ${jobId}:`, err);
  }
}

/** Release TTS weights from memory (no-op if TTS is not running). */
async function unloadTtsModel(): Promise<void> {
  if (!(await isTtsProcessAlive())) return;
  try {
    const res = await fetch(`${TTS_URL}/tts/unload`, { method: "POST" });
    if (!res.ok) {
      console.warn(`[TTS] unload HTTP ${res.status}`);
      return;
    }
    const body = (await res.json()) as { unloaded?: boolean };
    console.log(`[TTS] Model unload:`, body);
  } catch (err: any) {
    const msg = err?.message || String(err);
    if (/ConnectionRefused|ECONNREFUSED|Unable to connect/i.test(msg)) return;
    console.warn(`[TTS] Failed to unload model:`, err);
  }
}

function parsePageRange(startPage: unknown, endPage: unknown): { start: number; end: number } {
  const start = Math.max(1, parseInt(String(startPage ?? "1"), 10) || 1);
  let end = start; // default: only the start page/section (not “to end of document”)

  if (endPage !== null && endPage !== undefined && String(endPage).trim() !== "") {
    end = parseInt(String(endPage), 10);
    if (!Number.isFinite(end) || end < 1) {
      throw new Error("A página/seção final é inválida.");
    }
    if (end < start) {
      throw new Error(`A página/seção final (${end}) não pode ser menor que a inicial (${start}).`);
    }
  }
  return { start, end };
}

/** Join pdf.js text items into plain text, preserving line breaks. */
function textContentToPlainText(textContent: {
  items: Array<
    | {
        str?: string;
        transform?: number[];
        width?: number;
        height?: number;
        hasEOL?: boolean;
      }
    | unknown
  >;
}): string {
  const lines: string[] = [];
  let currentLine = "";
  let lastY: number | null = null;
  let lastEndX: number | null = null;
  let lastHeight = 10;

  for (const raw of textContent.items) {
    const item = raw as {
      str?: string;
      transform?: number[];
      width?: number;
      height?: number;
      hasEOL?: boolean;
    };
    if (typeof item?.str !== "string") continue;
    const str = item.str;

    const x = Array.isArray(item.transform) ? Number(item.transform[4]) : 0;
    const y = Array.isArray(item.transform) ? Number(item.transform[5]) : null;
    const fontSize = Array.isArray(item.transform)
      ? Math.abs(Number(item.transform[3] || item.transform[0]) || 0)
      : 0;
    const height = item.height || fontSize || lastHeight || 10;
    const width =
      item.width && item.width > 0 ? item.width : str.length * height * 0.45;

    const newLine =
      lastY !== null &&
      y !== null &&
      Math.abs(y - lastY) > Math.max(2.5, height * 0.35);

    if (newLine) {
      if (currentLine.trim()) lines.push(currentLine.trimEnd());
      currentLine = "";
      lastEndX = null;
    }

    if (currentLine && str) {
      const alreadySpaced = currentLine.endsWith(" ") || str.startsWith(" ");
      if (!alreadySpaced) {
        if (lastEndX != null && Number.isFinite(x)) {
          const gap = x - lastEndX;
          // Tracked/display letters sit closer than a real word space (~0.2em+).
          if (gap > height * 0.18) currentLine += " ";
        } else {
          currentLine += " ";
        }
      }
    }
    currentLine += str;

    if (item.hasEOL) {
      if (currentLine.trim()) lines.push(currentLine.trimEnd());
      currentLine = "";
      lastY = null;
      lastEndX = null;
      continue;
    }

    lastY = y;
    lastEndX = Number.isFinite(x) ? x + width : lastEndX;
    lastHeight = height;
  }

  if (currentLine.trim()) lines.push(currentLine.trimEnd());
  return lines.join("\n").trim();
}

function resolveTextRepairPython(): TextRepairPython | null {
  if (process.platform === "darwin") {
    const mlxDir = path.join(AURA_ROOT, "qwen3-tts-apple-silicon");
    const sitePackages = path.join(mlxDir, "site-packages");
    const bundled = resolveBundledPython(AURA_ROOT);
    if (bundled && fs.existsSync(sitePackages)) {
      return { bin: bundled.bin, home: bundled.home, pythonPath: sitePackages };
    }
    const venvPython = path.join(mlxDir, ".venv", "bin", "python");
    if (fs.existsSync(venvPython)) return { bin: venvPython };
  }

  const torchDir = path.join(AURA_ROOT, "tts", "torch");
  const torchSite = path.join(torchDir, "site-packages");
  const bundled = resolveBundledPython(AURA_ROOT);
  if (bundled && fs.existsSync(torchSite)) {
    return { bin: bundled.bin, home: bundled.home, pythonPath: torchSite };
  }
  const torchVenv = [
    path.join(torchDir, ".venv", "bin", "python"),
    path.join(torchDir, ".venv", "Scripts", "python.exe"),
  ].find((p) => fs.existsSync(p));
  if (torchVenv) return { bin: torchVenv };
  return null;
}

/** Local PDF text extraction via pdf.js (no network / API key). */
async function extractTextFromPdfLocal(
  pdfBase64: string,
  start: number,
  end: number
): Promise<{ text: string; actualStart: number; actualEnd: number; totalPages: number }> {
  const pdfjs = await import("pdfjs-dist/legacy/build/pdf.mjs");
  const data = new Uint8Array(Buffer.from(pdfBase64, "base64"));
  const loadingTask = pdfjs.getDocument({
    data,
    useSystemFonts: true,
  });
  const doc = await loadingTask.promise;
  const totalPages = doc.numPages;

  if (totalPages < 1) {
    throw new Error("O PDF não possui páginas legíveis.");
  }

  if (start > totalPages) {
    throw new Error(
      `O PDF possui apenas ${totalPages} página(s), mas a página inicial solicitada foi ${start}.`
    );
  }

  const actualStart = start;
  const actualEnd = Math.min(end, totalPages);
  if (end > totalPages) {
    console.warn(
      `[PDF Extract] Requested end page ${end} exceeds document (${totalPages}); clamping to ${totalPages}.`
    );
  }
  if (actualEnd < actualStart) {
    throw new Error("Intervalo de páginas inválido após o ajuste ao tamanho do PDF.");
  }

  const pageTexts: string[] = [];
  for (let pageNum = actualStart; pageNum <= actualEnd; pageNum++) {
    const page = await doc.getPage(pageNum);
    const content = await page.getTextContent();
    const pageText = textContentToPlainText(content);

    if (!pageText) {
      continue;
    }

    const { text: withoutPagination, strippedLeading } =
      stripPageEdgePagination(pageText);

    if (!withoutPagination) {
      continue;
    }

    if (pageTexts.length === 0) {
      pageTexts.push(withoutPagination);
      continue;
    }

    const glue = strippedLeading ? PAGE_NUMBER_GAP : "\n\n";
    pageTexts.push(`${glue}${withoutPagination}`);
  }

  return {
    text: pageTexts.join("").trim(),
    actualStart,
    actualEnd,
    totalPages,
  };
}

async function extractDocumentText(options: {
  fileData: string;
  fileType: "pdf" | "epub";
  start: number;
  end: number;
}): Promise<{ extractedText: string; pagesNarrated: string; actualStart: number; actualEnd: number }> {
  const { fileData, fileType, start, end } = options;
  const engine = activeTtsEngine();

  if (fileType === "pdf") {
    const extracted = await extractTextFromPdfLocal(fileData, start, end);
    let extractedText = sanitizeExtractedText(extracted.text, engine);
    extractedText = await repairExtractedTextWithModel(extractedText, {
      auraRoot: AURA_ROOT,
      dataDir: AURA_DATA_DIR,
      python: resolveTextRepairPython(),
    });
    extractedText = replaceStandaloneNumericLines(extractedText);
    extractedText = textForEngine(extractedText, engine);

    if (!extractedText) {
      throw new Error(
        "Não foi possível extrair texto das páginas selecionadas ou o PDF não possui conteúdo legível nesse intervalo."
      );
    }

    const pagesNarrated =
      extracted.actualStart === extracted.actualEnd
        ? `Página ${extracted.actualStart}`
        : `Páginas ${extracted.actualStart} - ${extracted.actualEnd}`;

    return {
      extractedText,
      pagesNarrated,
      actualStart: extracted.actualStart,
      actualEnd: extracted.actualEnd,
    };
  }

  // EPUB
  const tmpDir = os.tmpdir();
  const tempPath = path.join(
    tmpDir,
    `temp_epub_${Date.now()}_${Math.random().toString(36).substring(7)}.epub`
  );
  try {
    await fs.promises.writeFile(tempPath, Buffer.from(fileData, "base64"));
    let extractedText = await extractTextFromEpub(tempPath, start, end);
    extractedText = sanitizeExtractedText(extractedText, engine);
    extractedText = await repairExtractedTextWithModel(extractedText, {
      auraRoot: AURA_ROOT,
      dataDir: AURA_DATA_DIR,
      python: resolveTextRepairPython(),
    });
    extractedText = replaceStandaloneNumericLines(extractedText);
    extractedText = textForEngine(extractedText, engine);

    if (!extractedText) {
      throw new Error(
        "O EPUB não possui conteúdo legível nas seções/capítulos selecionados."
      );
    }
    const pagesNarrated =
      start === end
        ? `Capítulo/Seção ${start}`
        : `Capítulos/Seções ${start} - ${end}`;
    return {
      extractedText,
      pagesNarrated,
      actualStart: start,
      actualEnd: end,
    };
  } finally {
    await fs.promises.unlink(tempPath).catch(() => {});
  }
}

const PARAS_PER_CHUNK = 5;
const MIN_PARAGRAPH_WORDS = 20;
/**
 * Qwen + M5 NAX can drop the middle of long prompts (mlx-audio#464).
 * Keep each generate() to ~one short paragraph. Kokoro still uses 5 paras.
 */
const QWEN_MAX_CHUNK_CHARS = 280;

function paragraphWordCount(text: string): number {
  return text.trim().split(/\s+/).filter(Boolean).length;
}

/** Spoken lines that would sound odd if TTS saw them without the surrounding exchange. */
function isDialogueLine(text: string): boolean {
  return /^(?:[-–—―]|[“”"«»])/.test(text.trim());
}

function shouldAttachToPrevious(text: string, prev: string): boolean {
  if (paragraphWordCount(text) < MIN_PARAGRAPH_WORDS) return true;
  // Consecutive spoken lines stay in one prompt (question + reply, etc.).
  return isDialogueLine(text) && isDialogueLine(prev);
}

/**
 * Keep short lines / dialogue in the same TTS prompt as nearby text so the model
 * has speaker context, then emit every 5 remaining paragraphs as one chunk.
 * Qwen additionally splits oversized packs so long prompts do not drop the middle.
 */
function splitOversizedChunk(text: string, maxChars: number): string[] {
  const trailingNewlines = text.match(/\n+$/)?.[0] ?? "";
  const trimmed = text.trim();
  if (trimmed.length <= maxChars) {
    return [trimmed + trailingNewlines];
  }

  const sentences = trimmed.match(/[^.!?…]+[.!?…]+(?:\s+|$)|[^.!?…]+$/g) || [trimmed];
  const out: string[] = [];
  let current = "";

  const pushWords = (s: string) => {
    const words = s.split(/\s+/).filter(Boolean);
    let buf = "";
    for (const word of words) {
      const next = buf ? `${buf} ${word}` : word;
      if (next.length <= maxChars) {
        buf = next;
      } else {
        if (buf) out.push(buf);
        buf = word;
      }
    }
    if (buf) current = buf;
  };

  for (const sentence of sentences) {
    const s = sentence.trim();
    if (!s) {
      continue;
    }

    const next = current ? `${current} ${s}` : s;
    if (next.length <= maxChars) {
      current = next;
      continue;
    }

    if (current) {
      out.push(current);
    }

    current = "";
    if (s.length <= maxChars) {
      current = s;
    } else {
      pushWords(s);
    }
  }

  if (current) {
    out.push(current);
  }
  const parts = out.length > 0 ? out : [trimmed];

  if (trailingNewlines && parts.length > 0) {
    parts[parts.length - 1] += trailingNewlines;
  }

  return parts;
}

function appendLogicalParagraphs(source: string, logical: string[]): void {
  for (const para of source.split(/\n+/)) {
    const trimmed = para.trim();
    if (!trimmed) {
      continue;
    }

    if (isPauseLine(trimmed)) {
      logical.push(trimmed);
      continue;
    }

    const prev = logical[logical.length - 1];
    if (
      prev &&
      prev !== QWEN_BREAK_GAP &&
      !isPauseLine(prev) &&
      shouldAttachToPrevious(trimmed, prev)
    ) {
      logical[logical.length - 1] = `${prev}\n${trimmed}`;
    } else {
      logical.push(trimmed);
    }
  }
}

function splitTextIntoChunks(
  text: string,
  engine: TtsEngineId = activeTtsEngine()
): string[] {
  const logical: string[] = [];

  if (engine === "qwen3") {
    const sections = replaceBreakTagsWithNewlines(text).split(/\n{3,}/);
    for (let i = 0; i < sections.length; i++) {
      appendLogicalParagraphs(sections[i], logical);
      if (
        i < sections.length - 1 &&
        logical.length > 0 &&
        logical[logical.length - 1] !== QWEN_BREAK_GAP
      ) {
        logical.push(QWEN_BREAK_GAP);
      }
    }
  } else {
    appendLogicalParagraphs(text, logical);
  }

  const chunks: string[] = [];
  let group: string[] = [];

  const flushGroup = () => {
    if (group.length === 0) {
      return;
    }

    chunks.push(group.join("\n"));
    group = [];
  };

  for (const item of logical) {
    if (item === QWEN_BREAK_GAP) {
      flushGroup();
      if (chunks.length > 0) {
        chunks[chunks.length - 1] += QWEN_BREAK_GAP;
      }

      continue;
    }

    if (isPauseLine(item)) {
      flushGroup();
      chunks.push(item);
      continue;
    }

    group.push(item);
    if (group.length >= PARAS_PER_CHUNK) {
      flushGroup();
    }
  }
  flushGroup();

  const merged: string[] = [];
  for (const chunk of chunks) {
    if (isPauseLine(chunk)) {
      merged.push(chunk);
      continue;
    }
    const last = merged[merged.length - 1];
    if (
      last &&
      !isPauseLine(last) &&
      paragraphWordCount(chunk) < MIN_PARAGRAPH_WORDS
    ) {
      const joiner = last.endsWith(QWEN_BREAK_GAP) ? "" : "\n";
      merged[merged.length - 1] = `${last}${joiner}${chunk}`;
    } else {
      merged.push(chunk);
    }
  }
  let packed = merged;
  if (
    merged.length >= 2 &&
    !isPauseLine(merged[0]) &&
    paragraphWordCount(merged[0]) < MIN_PARAGRAPH_WORDS &&
    !isPauseLine(merged[1])
  ) {
    const [first, second, ...rest] = merged;
    const joiner = first.endsWith(QWEN_BREAK_GAP) ? "" : "\n";
    packed = [`${first}${joiner}${second}`, ...rest];
  }
  if (engine === "qwen3") {
    return packed.flatMap((c) =>
      isPauseLine(c) ? [c] : splitOversizedChunk(c, QWEN_MAX_CHUNK_CHARS)
    );
  }
  return packed;
}

type NarrationArtifact = {
  /** Primary audio file (mp3 or m4b). */
  mp3Path: string;
  pcmPath: string;
  fileName: string;
  createdAt: number;
  format?: "mp3" | "m4b";
  coverPath?: string | null;
  mimeType?: string;
};

const narrationArtifacts = new Map<string, NarrationArtifact>();
const NARRATION_TMP_DIR = path.join(AURA_DATA_DIR, "tmp", "narration");
const CHUNK_CACHE_DIR = path.join(AURA_DATA_DIR, "tmp", "chunk-cache");

function ensureNarrationTmpDir() {
  fs.mkdirSync(NARRATION_TMP_DIR, { recursive: true });
}

function newNarrationId(): string {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

type ChunkCacheMeta = {
  docId: string;
  voice: string;
  engine: string;
  fingerprint: string;
  totalChunks: number;
  completedIndices: number[];
  sampleRate: number;
  updatedAt: number;
};

function isSafeDocId(docId: string): boolean {
  if (
    /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(
      docId
    )
  ) {
    return true;
  }
  // Legacy / short ids — path-safe only
  return /^[A-Za-z0-9_-]{6,80}$/.test(docId);
}

function chunkCacheDirFor(docId: string): string {
  return path.join(CHUNK_CACHE_DIR, docId);
}

function chunkPcmPath(docId: string, index: number): string {
  return path.join(chunkCacheDirFor(docId), `${String(index).padStart(4, "0")}.pcm`);
}

function chunkMetaPath(docId: string): string {
  return path.join(chunkCacheDirFor(docId), "meta.json");
}

function narrationFingerprint(
  text: string,
  voice: string,
  engine: string,
  language: string
): string {
  return createHash("sha256")
    .update(
      `${engine}\n${voice}\n${language}\n${VOICE_PREVIEW_CACHE_VERSION}\n` +
        `t=${TTS_TEMPERATURE}\n${text}`
    )
    .digest("hex")
    .slice(0, 40);
}

async function readChunkCacheMeta(docId: string): Promise<ChunkCacheMeta | null> {
  try {
    const raw = await fs.promises.readFile(chunkMetaPath(docId), "utf8");
    const meta = JSON.parse(raw) as ChunkCacheMeta;
    if (!meta || meta.docId !== docId) return null;
    return meta;
  } catch {
    return null;
  }
}

async function writeChunkCacheMeta(meta: ChunkCacheMeta): Promise<void> {
  const dir = chunkCacheDirFor(meta.docId);
  await fs.promises.mkdir(dir, { recursive: true });
  await fs.promises.writeFile(chunkMetaPath(meta.docId), JSON.stringify(meta, null, 2), "utf8");
}

async function clearChunkCache(docId: string): Promise<boolean> {
  if (!isSafeDocId(docId)) return false;
  const dir = chunkCacheDirFor(docId);
  try {
    await fs.promises.rm(dir, { recursive: true, force: true });
    return true;
  } catch {
    return false;
  }
}

async function ensureChunkCache(
  docId: string,
  voice: string,
  engine: string,
  fingerprint: string,
  totalChunks: number,
  sampleRate: number
): Promise<ChunkCacheMeta> {
  const existing = await readChunkCacheMeta(docId);
  if (
    existing &&
    existing.fingerprint === fingerprint &&
    existing.voice === voice &&
    existing.engine === engine &&
    existing.totalChunks === totalChunks
  ) {
    return existing;
  }
  await clearChunkCache(docId);
  const meta: ChunkCacheMeta = {
    docId,
    voice,
    engine,
    fingerprint,
    totalChunks,
    completedIndices: [],
    sampleRate,
    updatedAt: Date.now(),
  };
  await writeChunkCacheMeta(meta);
  return meta;
}

async function saveChunkPcm(
  docId: string,
  index: number,
  pcm: Buffer,
  meta: ChunkCacheMeta
): Promise<ChunkCacheMeta> {
  const dir = chunkCacheDirFor(docId);
  await fs.promises.mkdir(dir, { recursive: true });
  await fs.promises.writeFile(chunkPcmPath(docId, index), pcm);
  if (!meta.completedIndices.includes(index)) {
    meta.completedIndices = [...meta.completedIndices, index].sort((a, b) => a - b);
  }
  meta.updatedAt = Date.now();
  await writeChunkCacheMeta(meta);
  return meta;
}

async function concatCachedChunksToPcm(
  docId: string,
  totalChunks: number,
  outPath: string
): Promise<{ bytes: number; parts: number }> {
  const out = await fs.promises.open(outPath, "w");
  let bytes = 0;
  let parts = 0;
  try {
    for (let i = 0; i < totalChunks; i++) {
      const chunkPath = chunkPcmPath(docId, i);
      try {
        const pcm = await fs.promises.readFile(chunkPath);
        if (pcm.length === 0) continue;
        await out.write(pcm);
        bytes += pcm.length;
        parts += 1;
      } catch {
        throw new Error(`Bloco ${i + 1} ausente no cache de narração.`);
      }
    }
  } finally {
    await out.close();
  }
  return { bytes, parts };
}

async function chunkCacheStatus(docId: string): Promise<{
  exists: boolean;
  completed: number;
  total: number;
  voice: string | null;
  engine: string | null;
} | null> {
  if (!isSafeDocId(docId)) return null;
  const meta = await readChunkCacheMeta(docId);
  if (!meta) {
    return { exists: false, completed: 0, total: 0, voice: null, engine: null };
  }
  return {
    exists: meta.completedIndices.length > 0,
    completed: meta.completedIndices.length,
    total: meta.totalChunks,
    voice: meta.voice,
    engine: meta.engine,
  };
}

async function cleanupNarrationArtifact(id: string) {
  const art = narrationArtifacts.get(id);
  if (!art) return;
  narrationArtifacts.delete(id);
  await fs.promises.unlink(art.mp3Path).catch(() => undefined);
  await fs.promises.unlink(art.pcmPath).catch(() => undefined);
  if (art.coverPath) await fs.promises.unlink(art.coverPath).catch(() => undefined);
}

/** WAV + transcript used as voice-reference (ref_audio / ref_text for ICL). */
const VOICE_PREVIEW_DIR = path.join(AURA_ROOT, "assets", "voice-previews");
/** In-memory WAV previews (base64) layered on disk cache. Key: voice+locale. */
const voicePreviewCache = new Map<string, string>();

function voicePreviewSafeKey(voiceKey: string): string {
  return voiceKey.replace(/[^a-z0-9_-]/gi, "_").toLowerCase();
}

function voicePreviewFileKey(voiceName: string, languageId: string): string {
  const voice = voicePreviewSafeKey(voiceName);
  const lang = voicePreviewSafeKey(voicePreviewCacheTag(languageId));

  return `${voice}_${lang}_${VOICE_PREVIEW_CACHE_VERSION}`;
}

function voicePreviewWavPath(fileKey: string): string {
  return path.join(VOICE_PREVIEW_DIR, `${fileKey}.wav`);
}

function voicePreviewTextPath(fileKey: string): string {
  return path.join(VOICE_PREVIEW_DIR, `${fileKey}.txt`);
}

/** Drop trailing near-silence so truncated/greedy generations don't pad the preview. */
function trimTrailingSilence(samples: Int16Array, sampleRate = 24000): Int16Array {
  const threshold = 200;
  const padSamples = Math.floor(sampleRate * 0.12); // keep a short natural tail
  let last = samples.length - 1;
  while (last > 0 && Math.abs(samples[last]) <= threshold) last--;
  const end = Math.min(samples.length, last + 1 + padSamples);
  if (end >= samples.length) {
    return samples;
  }

  if (end < sampleRate * 0.4) {
    return samples;
  }

  return samples.subarray(0, end);
}

/** Encode mono PCM s16le as a WAV (usable as Qwen ICL ref_audio). */
function encodePcmToWav(samples: Int16Array, sampleRate = 24000): Buffer {
  const numChannels = 1;
  const bitsPerSample = 16;
  const blockAlign = (numChannels * bitsPerSample) / 8;
  const byteRate = sampleRate * blockAlign;
  const dataSize = samples.length * 2;
  const buffer = Buffer.alloc(44 + dataSize);

  buffer.write("RIFF", 0);
  buffer.writeUInt32LE(36 + dataSize, 4);
  buffer.write("WAVE", 8);
  buffer.write("fmt ", 12);
  buffer.writeUInt32LE(16, 16); // PCM fmt chunk size
  buffer.writeUInt16LE(1, 20); // PCM
  buffer.writeUInt16LE(numChannels, 22);
  buffer.writeUInt32LE(sampleRate, 24);
  buffer.writeUInt32LE(byteRate, 28);
  buffer.writeUInt16LE(blockAlign, 32);
  buffer.writeUInt16LE(bitsPerSample, 34);
  buffer.write("data", 36);
  buffer.writeUInt32LE(dataSize, 40);
  buffer.set(new Uint8Array(samples.buffer, samples.byteOffset, dataSize), 44);
  return buffer;
}

async function loadVoicePreviewFromDisk(fileKey: string): Promise<string | null> {
  const filePath = voicePreviewWavPath(fileKey);
  try {
    const buf = await fs.promises.readFile(filePath);
    if (buf.length > 0) {
      return buf.toString("base64");
    }
  } catch {
    // miss
  }

  return null;
}

async function saveVoicePreviewToDisk(
  fileKey: string,
  wav: Buffer,
  transcript: string
): Promise<void> {
  await fs.promises.mkdir(VOICE_PREVIEW_DIR, { recursive: true });
  const wavPath = voicePreviewWavPath(fileKey);
  await fs.promises.writeFile(wavPath, wav);
  // Transcript must match the WAV for ICL voice reference (ref_text).
  await fs.promises.writeFile(voicePreviewTextPath(fileKey), `${transcript.trim()}\n`, "utf8");
  console.log(
    `[VoicePreview] Áudio salvo: nome="${path.basename(wavPath)}" caminho="${path.resolve(wavPath)}"`
  );
}

/** Remove disk WAV+TXT and in-memory preview for one voice+locale. */
async function deleteVoicePreviewFiles(
  voiceName: string,
  languageId: string
): Promise<{ deleted: boolean; fileKey: string }> {
  const fileKey = voicePreviewFileKey(voiceName, languageId);
  voicePreviewCache.delete(fileKey);

  let deleted = false;
  for (const filePath of [voicePreviewWavPath(fileKey), voicePreviewTextPath(fileKey)]) {
    try {
      await fs.promises.unlink(filePath);
      deleted = true;
      console.log(`[VoicePreview] Removido: ${path.basename(filePath)}`);
    } catch {
      // miss
    }
  }

  return { deleted, fileKey };
}

/**
 * Ensure a disk WAV+TXT voice anchor exists for ICL narration.
 * One sample per speaker+locale; skipIcl generate uses that locale's preview text.
 */
async function ensureVoicePreview(
  voiceName: string,
  languageId: string
): Promise<{ refAudioPath: string; refText: string; created: boolean }> {
  const language = resolveNarrationLanguageId(languageId);
  const previewText = previewTextForLanguage(language);
  const ttsLanguage = languagePayloadForTts(
    language,
    activeTtsEngine(),
    readKokoroBackend(AURA_DATA_DIR)
  );
  const fileKey = voicePreviewFileKey(voiceName, language);
  const wavPath = voicePreviewWavPath(fileKey);
  const txtPath = voicePreviewTextPath(fileKey);

  try {
    await fs.promises.access(wavPath, fs.constants.R_OK);
    await fs.promises.access(txtPath, fs.constants.R_OK);
    const fromDisk = (await fs.promises.readFile(txtPath, "utf8")).trim();
    // Stale transcript (preview copy changed without a version bump) → regenerate.
    if (fromDisk && fromDisk !== previewText.trim()) {
      console.log(
        `[VoicePreview] Transcript mismatch for ${fileKey}; regenerating ICL anchor.`
      );
    } else {
      const refText = fromDisk || previewText;
      const wavBuf = await fs.promises.readFile(wavPath);

      if (wavBuf.length > 0) {
        voicePreviewCache.set(fileKey, wavBuf.toString("base64"));
        console.log(
          `[VoicePreview] Using locale anchor: ${path.basename(wavPath)} (lang=${language})`
        );

        return { refAudioPath: wavPath, refText, created: false };
      }
    }
  } catch {
    // miss — generate below
  }

  console.log(
    `[VoicePreview] Generating ICL anchor for ${voiceName} (${language})...`
  );
  const { pcm, sampleRate, cancelled } = await synthesizeWithTts(
    previewText,
    voiceName,
    undefined,
    undefined,
    {
      skipIcl: true,
      language: ttsLanguage,
      instruct: mergeQwenInstruct(language, TTS_INSTRUCT),
    }
  );
  if (cancelled || pcm.length === 0) {
    throw new Error(
      `Não foi possível gerar a âncora de voz para "${voiceName}". ` +
        "Toque a prévia na UI ou verifique o servidor Qwen TTS."
    );
  }

  const samples = trimTrailingSilence(
    new Int16Array(pcm.buffer, pcm.byteOffset, Math.floor(pcm.byteLength / 2)),
    sampleRate
  );
  const wavBuffer = encodePcmToWav(samples, sampleRate);
  voicePreviewCache.set(fileKey, wavBuffer.toString("base64"));
  await saveVoicePreviewToDisk(fileKey, wavBuffer, previewText);

  return {
    refAudioPath: wavPath,
    refText: previewText,
    created: true,
  };
}

// Short sample of a narrator voice (WAV base64 + disk WAV/TXT for voice reference)
app.post("/api/voice-preview", async (req, res) => {
  let didGenerate = false;
  try {
    const engine = activeTtsEngine();
    const languageId = resolveNarrationLanguageIdFromRequest(req.body?.language);
    const previewText = previewTextForLanguage(languageId);
    const ttsLanguage = languagePayloadForTts(
      languageId,
      engine,
      readKokoroBackend(AURA_DATA_DIR)
    );
    const voiceName =
      String(req.body?.voiceName || "").trim() ||
      (engine === "kokoro" ? "af_heart" : "Vivian");
    const fileKey = voicePreviewFileKey(voiceName, languageId);

    const memCached = voicePreviewCache.get(fileKey);
    if (memCached) {
      return res.json({
        audioData: memCached,
        mimeType: "audio/wav",
        voiceName,
        language: languageId,
        cached: true,
        source: "memory",
        filePath: voicePreviewWavPath(fileKey),
        textPath: voicePreviewTextPath(fileKey),
        engine,
      });
    }

    const diskCached = await loadVoicePreviewFromDisk(fileKey);
    if (diskCached) {
      voicePreviewCache.set(fileKey, diskCached);

      return res.json({
        audioData: diskCached,
        mimeType: "audio/wav",
        voiceName,
        language: languageId,
        cached: true,
        source: "disk",
        filePath: voicePreviewWavPath(fileKey),
        textPath: voicePreviewTextPath(fileKey),
        engine,
      });
    }

    // Kokoro: synthesize a short clip and persist it to disk (no ICL, so the
    // saved WAV is just a reusable preview rather than a voice-clone anchor).
    if (engine === "kokoro") {
      const { pcm, sampleRate } = await synthesizeWithTts(
        previewText,
        voiceName,
        undefined,
        undefined,
        { skipIcl: true, language: ttsLanguage, instruct: mergeQwenInstruct(languageId, TTS_INSTRUCT) }
      );
      didGenerate = true;
      if (pcm.length === 0) {
        throw new Error("Prévia Kokoro vazia.");
      }

      const samples = trimTrailingSilence(
        new Int16Array(pcm.buffer, pcm.byteOffset, Math.floor(pcm.byteLength / 2)),
        sampleRate
      );
      const wavBuffer = encodePcmToWav(samples, sampleRate);
      const audioData = wavBuffer.toString("base64");
      voicePreviewCache.set(fileKey, audioData);
      await saveVoicePreviewToDisk(fileKey, wavBuffer, previewText);

      return res.json({
        audioData,
        mimeType: "audio/wav",
        voiceName,
        language: languageId,
        cached: false,
        source: "generated",
        filePath: voicePreviewWavPath(fileKey),
        textPath: voicePreviewTextPath(fileKey),
        engine,
      });
    }

    const anchor = await ensureVoicePreview(voiceName, languageId);
    didGenerate = anchor.created;
    const audioData =
      voicePreviewCache.get(fileKey) ||
      (await fs.promises.readFile(anchor.refAudioPath)).toString("base64");
    voicePreviewCache.set(fileKey, audioData);

    return res.json({
      audioData,
      mimeType: "audio/wav",
      voiceName,
      language: languageId,
      cached: false,
      source: "generated",
      filePath: anchor.refAudioPath,
      textPath: voicePreviewTextPath(fileKey),
      engine,
    });
  } catch (err: any) {
    console.error("[VoicePreview]", err?.message || err);
    didGenerate = true;

    return res.status(500).json({
      error: err?.message || "Falha ao gerar prévia de voz.",
    });
  } finally {
    if (didGenerate) {
      await unloadTtsModel();
    }

  }
});

app.delete("/api/voice-preview", async (req, res) => {
  try {
    const languageId = resolveNarrationLanguageIdFromRequest(req.body?.language);
    const voiceName = String(req.body?.voiceName || "").trim();

    if (!voiceName) {
      return res.status(400).json({ error: "O nome da voz é obrigatório." });
    }

    const { deleted, fileKey } = await deleteVoicePreviewFiles(voiceName, languageId);

    return res.json({
      voiceName,
      language: languageId,
      deleted,
      fileKey,
      engine: activeTtsEngine(),
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Falha ao excluir prévia de voz.";

    return res.status(500).json({ error: message });
  }
});

// Active TTS engine (qwen3 | kokoro)
app.get("/api/tts-engine", (_req, res) => {
  try {
    const status = getModelsStatus();
    res.json({
      engine: status.engine,
      engines: status.engines,
      voices: status.voices,
      ready: status.ready,
      kokoroDevice: status.kokoroDevice,
      kokoroBackend: status.kokoroBackend,
      kokoroAccel: status.kokoroAccel,
    });
  } catch (err: any) {
    res.status(500).json({ error: err?.message || String(err) });
  }
});

app.post("/api/tts-engine", async (req, res) => {
  try {
    const nextEngine = req.body?.engine;
    const nextDevice = req.body?.kokoroDevice;
    const nextBackend = req.body?.kokoroBackend;
    const hasEngine = nextEngine !== undefined && nextEngine !== null;
    const hasDevice = nextDevice !== undefined && nextDevice !== null;
    const hasBackend = nextBackend !== undefined && nextBackend !== null;

    if (hasEngine && !isTtsEngineId(nextEngine)) {
      return res.status(400).json({ error: 'engine must be "qwen3" or "kokoro"' });
    }
    if (hasDevice && !isKokoroDeviceId(nextDevice)) {
      return res.status(400).json({ error: 'kokoroDevice must be "cpu" or "gpu"' });
    }
    if (hasBackend && !isKokoroBackendId(nextBackend)) {
      return res.status(400).json({ error: 'kokoroBackend must be "mlx" or "onnx"' });
    }
    if (hasBackend && nextBackend === "mlx" && process.platform !== "darwin") {
      return res.status(400).json({ error: "MLX Kokoro is available only on macOS" });
    }
    if (!hasEngine && !hasDevice && !hasBackend) {
      return res.status(400).json({
        error: 'Provide "engine", "kokoroDevice" and/or "kokoroBackend"',
      });
    }

    const prevEngine = activeTtsEngine();
    const prevDevice = readKokoroDevice(AURA_DATA_DIR);
    const prevBackend = readKokoroBackend(AURA_DATA_DIR);

    if (hasEngine) writeTtsEngine(AURA_DATA_DIR, nextEngine);
    if (hasDevice) writeKokoroDevice(AURA_DATA_DIR, nextDevice);
    if (hasBackend) writeKokoroBackend(AURA_DATA_DIR, nextBackend);

    const engineChanged = hasEngine && prevEngine !== nextEngine;
    const deviceChanged = hasDevice && prevDevice !== nextDevice;
    const backendChanged = hasBackend && prevBackend !== nextBackend;
    // Restart TTS if engine changed, or Kokoro configuration changed while active.
    const needRestart =
      engineChanged ||
      ((deviceChanged || backendChanged) &&
        (hasEngine ? nextEngine === "kokoro" : prevEngine === "kokoro"));

    if (needRestart) {
      await unloadTtsModel().catch(() => undefined);
      stopManagedTts();
    }

    const status = getModelsStatus();
    res.json({
      engine: status.engine,
      engines: status.engines,
      voices: status.voices,
      ready: status.ready,
      kokoroDevice: status.kokoroDevice,
      kokoroBackend: status.kokoroBackend,
      kokoroAccel: status.kokoroAccel,
      restarted: needRestart,
    });
  } catch (err: any) {
    res.status(500).json({ error: err?.message || String(err) });
  }
});

/** Precompile Kokoro MIGraphX graphs for several input lengths (GPU only). */
app.get("/api/tts/kokoro-warmup", async (_req, res) => {
  try {
    if (activeTtsEngine() !== "kokoro") {
      return res.json({
        running: false,
        done: false,
        skipped: true,
        message: "Motor ativo não é Kokoro.",
      });
    }
    if (!(await isTtsProcessReady("kokoro"))) {
      return res.json({
        running: false,
        done: false,
        serverReady: false,
        phase: "Servidor Kokoro ainda não iniciado.",
      });
    }
    const ttsRes = await fetch(`${TTS_URL}/tts/warmup`);
    const body = await ttsRes.json();
    if (!ttsRes.ok) {
      return res.status(ttsRes.status).json(body);
    }
    res.json(body);
  } catch (err: any) {
    res.status(500).json({ error: err?.message || String(err) });
  }
});

app.post("/api/tts/kokoro-warmup", async (req, res) => {
  try {
    if (activeTtsEngine() !== "kokoro") {
      return res.status(400).json({ error: "Selecione o motor Kokoro antes de aquecer a GPU." });
    }
    if (readKokoroDevice(AURA_DATA_DIR) !== "gpu") {
      return res.status(400).json({
        error: "Warm-up só se aplica com Aceleração Kokoro = GPU.",
      });
    }
    // Compiles can take many minutes — allow a long client wait if they poll status instead.
    await ensureTtsRunning(120_000);
    const ttsRes = await fetch(`${TTS_URL}/tts/warmup`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ voice: req.body?.voice || undefined }),
    });
    const body = await ttsRes.json();
    if (!ttsRes.ok) {
      return res.status(ttsRes.status).json(body);
    }
    res.json(body);
  } catch (err: any) {
    res.status(500).json({ error: err?.message || String(err) });
  }
});

// TTS model install status (Base + CustomVoice)
app.get("/api/models/status", (_req, res) => {
  try {
    res.json(getModelsStatus());
  } catch (err: any) {
    res.status(500).json({ ready: false, error: err?.message || String(err) });
  }
});

// Download missing models; streams NDJSON progress events (TS/fetch, not Python)
app.post("/api/models/download", async (req, res) => {
  if (isModelDownloadActive()) {
    res.status(409).json({ error: "Download já em andamento." });
    return;
  }

  const status = getModelsStatus();
  if (status.ready) {
    res.json({ ready: true, skipped: true, modelsDir: status.modelsDir });
    return;
  }

  res.setHeader("Content-Type", "application/x-ndjson; charset=utf-8");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  if (typeof (res as any).flushHeaders === "function") {
    (res as any).flushHeaders();
  }

  const writeEvent = (payload: Record<string, unknown>) => {
    if (!res.writableEnded) {
      res.write(`${JSON.stringify(payload)}\n`);
    }
  };

  try {
    await downloadMissingModels({
      auraRoot: AURA_ROOT,
      auraDataDir: AURA_DATA_DIR,
      onEvent: writeEvent,
    });
  } catch (err: any) {
    if (!res.writableEnded) {
      writeEvent({
        type: "error",
        message: err?.message || String(err),
        ready: getModelsStatus().ready,
      });
    }
  } finally {
    if (!res.writableEnded) res.end();
  }
});

app.post("/api/models/download/cancel", (_req, res) => {
  const cancelled = cancelModelDownload();
  res.json({
    success: cancelled,
    message: cancelled ? "Cancelamento solicitado." : "Nenhum download em andamento.",
  });
});

app.delete("/api/models", async (req, res) => {
  try {
    if (isModelDownloadActive()) {
      return res.status(409).json({ error: "Cancele o download antes de excluir modelos." });
    }
    // Free GPU/RAM if TTS was using the weights
    stopManagedTts();
    await unloadTtsModel().catch(() => undefined);

    const idsRaw = req.query.id;
    const ids = Array.isArray(idsRaw)
      ? idsRaw.map(String)
      : typeof idsRaw === "string" && idsRaw.length
        ? idsRaw.split(",").map((s) => s.trim()).filter(Boolean)
        : undefined;

    const result = deleteModels(AURA_ROOT, AURA_DATA_DIR, ids);
    res.json({
      success: true,
      ...result,
      status: getModelsStatus(),
    });
  } catch (err: any) {
    res.status(500).json({ error: err?.message || String(err) });
  }
});

// Health Check API
app.get("/api/health", async (req, res) => {
  let ttsReady = false;
  let modelLoaded = false;
  const models = getModelsStatus();
  try {
    const ttsRes = await fetch(`${TTS_URL}/health`);
    if (ttsRes.ok) {
      const body = (await ttsRes.json()) as { ready?: boolean; modelLoaded?: boolean };
      ttsReady = body.ready !== false;
      modelLoaded = !!body.modelLoaded;
    }
  } catch {
    ttsReady = false;
    modelLoaded = false;
  }
  res.json({
    status: "ok",
    time: new Date().toISOString(),
    models,
    tts: {
      provider: models.engine === "kokoro" ? "kokoro" : "qwen3-tts",
      backend: models.backend,
      engine: models.engine,
      url: TTS_URL,
      ready: ttsReady,
      modelLoaded,
    },
  });
});

// Active generation tasks map for cancellation/stop support
const activeTasks = new Map<string, { stopped: boolean; abort: AbortController }>();

function resolveDownloadsDir(): string {
  const home = os.homedir();
  const candidates =
    process.platform === "win32"
      ? [
          process.env.USERPROFILE ? path.join(process.env.USERPROFILE, "Downloads") : "",
          path.join(home, "Downloads"),
          path.join(home, "Transferências"),
        ]
      : [
          path.join(home, "Downloads"),
          path.join(home, "Transferências"), // common PT-BR folder name on macOS/Linux
        ];

  for (const candidate of candidates) {
    if (candidate && fs.existsSync(candidate)) return candidate;
  }
  // Prefer creating the English "Downloads" name if nothing exists yet
  const fallback = path.join(home, "Downloads");
  fs.mkdirSync(fallback, { recursive: true });
  return fallback;
}

function sanitizeDownloadBaseName(name: string): string {
  const cleaned = String(name || "narracao")
    .replace(/[<>:"/\\|?*\u0000-\u001f]/g, "_")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/^\.+/, "")
    .slice(0, 180);
  return cleaned || "narracao";
}

function splitDownloadName(baseName: string): { stem: string; ext: string } {
  const raw = String(baseName || "narracao.mp3").trim() || "narracao.mp3";
  const m = raw.match(/^(.+?)(\.[a-z0-9]+)?$/i);
  const stem = sanitizeDownloadBaseName((m?.[1] || "narracao").replace(/\.(mp3|m4b|jpg|jpeg)$/i, ""));
  let ext = (m?.[2] || ".mp3").toLowerCase();
  if (!ext.startsWith(".")) ext = `.${ext}`;
  if (![".mp3", ".m4b", ".jpg", ".jpeg"].includes(ext)) ext = ".mp3";
  if (ext === ".jpeg") ext = ".jpg";
  return { stem, ext };
}

function uniqueDownloadPath(dir: string, baseName: string): string {
  const { stem, ext } = splitDownloadName(baseName);
  let candidate = path.join(dir, `${stem}${ext}`);
  if (!fs.existsSync(candidate)) return candidate;

  for (let i = 1; i < 1000; i++) {
    candidate = path.join(dir, `${stem} (${i})${ext}`);
    if (!fs.existsSync(candidate)) return candidate;
  }
  return path.join(dir, `${stem}-${Date.now()}${ext}`);
}

async function copyToDownloads(
  srcPath: string,
  fileName: string
): Promise<{ path: string; fileName: string; directory: string }> {
  const downloadsDir = resolveDownloadsDir();
  const destPath = uniqueDownloadPath(downloadsDir, fileName);
  await fs.promises.copyFile(srcPath, destPath);
  return {
    path: destPath,
    fileName: path.basename(destPath),
    directory: downloadsDir,
  };
}

// Save completed audio (and optional cover) into the user's Downloads folder
app.post("/api/save-to-downloads", async (req, res) => {
  try {
    const { audioData, audioId, fileName, coverData, coverFileName, saveCover } = req.body as {
      audioData?: string;
      audioId?: string;
      fileName?: string;
      coverData?: string;
      coverFileName?: string;
      saveCover?: boolean;
    };

    const downloadsDir = resolveDownloadsDir();
    const saved: { path: string; fileName: string }[] = [];

    if (audioId && typeof audioId === "string") {
      const art = narrationArtifacts.get(audioId);
      if (!art || !fs.existsSync(art.mp3Path)) {
        return res.status(404).json({ error: "Áudio não encontrado (expirou ou id inválido)." });
      }
      const fmt = art.format || "mp3";
      const audioName =
        fileName ||
        `${sanitizeDownloadBaseName(art.fileName)}.${fmt}`;
      const audioSaved = await copyToDownloads(art.mp3Path, audioName);
      saved.push(audioSaved);
      console.log(`[SaveDownloads] Copied audio → ${audioSaved.path}`);

      let coverSaved: { path: string; fileName: string } | null = null;
      // M4B already embeds artwork — skip writing a separate cover JPEG.
      if (
        saveCover !== false &&
        fmt !== "m4b" &&
        art.coverPath &&
        fs.existsSync(art.coverPath)
      ) {
        const { stem } = splitDownloadName(audioSaved.fileName);
        const coverName = coverFileName || `${stem}.jpg`;
        coverSaved = await copyToDownloads(art.coverPath, coverName);
        saved.push(coverSaved);
        console.log(`[SaveDownloads] Copied cover → ${coverSaved.path}`);
      }

      return res.json({
        success: true,
        path: audioSaved.path,
        fileName: audioSaved.fileName,
        coverFileName: coverSaved?.fileName ?? null,
        coverPath: coverSaved?.path ?? null,
        directory: downloadsDir,
        files: saved,
      });
    }

    if (!audioData || typeof audioData !== "string") {
      return res.status(400).json({ error: "audioId ou audioData (base64) é obrigatório." });
    }

    const buffer = Buffer.from(audioData, "base64");
    if (buffer.length === 0) {
      return res.status(400).json({ error: "Áudio vazio — nada para salvar." });
    }

    const destPath = uniqueDownloadPath(downloadsDir, fileName || "narracao.mp3");
    await fs.promises.writeFile(destPath, buffer);
    saved.push({ path: destPath, fileName: path.basename(destPath) });
    console.log(`[SaveDownloads] Saved ${buffer.length} bytes → ${destPath}`);

    let coverSaved: { path: string; fileName: string } | null = null;
    if (coverData && typeof coverData === "string") {
      const coverBuf = Buffer.from(coverData, "base64");
      if (coverBuf.length > 0) {
        const { stem } = splitDownloadName(path.basename(destPath));
        const coverDest = uniqueDownloadPath(downloadsDir, coverFileName || `${stem}.jpg`);
        await fs.promises.writeFile(coverDest, coverBuf);
        coverSaved = { path: coverDest, fileName: path.basename(coverDest) };
        saved.push(coverSaved);
      }
    }

    return res.json({
      success: true,
      path: destPath,
      fileName: path.basename(destPath),
      coverFileName: coverSaved?.fileName ?? null,
      coverPath: coverSaved?.path ?? null,
      directory: downloadsDir,
      files: saved,
    });
  } catch (err: any) {
    console.error("[SaveDownloads] Failed:", err);
    return res.status(500).json({
      error: err?.message || "Não foi possível salvar o áudio em Downloads.",
    });
  }
});

/** Stream a narration audio file from disk. */
app.get("/api/narration-audio/:id", async (req, res) => {
  try {
    const id = String(req.params.id || "");
    const art = narrationArtifacts.get(id);
    if (!art || !fs.existsSync(art.mp3Path)) {
      return res.status(404).json({ error: "Áudio não encontrado." });
    }
    const fmt = art.format || "mp3";
    const mime =
      art.mimeType || (fmt === "m4b" ? "audio/mp4" : "audio/mpeg");
    const st = await fs.promises.stat(art.mp3Path);
    res.setHeader("Content-Type", mime);
    res.setHeader("Content-Length", String(st.size));
    res.setHeader(
      "Content-Disposition",
      `inline; filename="${sanitizeDownloadBaseName(art.fileName)}.${fmt}"`
    );
    const stream = fs.createReadStream(art.mp3Path);
    stream.on("error", (err) => {
      console.error("[NarrationAudio] stream error:", err);
      if (!res.headersSent) res.status(500).end();
      else res.end();
    });
    stream.pipe(res);
  } catch (err: any) {
    res.status(500).json({ error: err?.message || "Falha ao servir áudio." });
  }
});

app.get("/api/narration-cover/:id", async (req, res) => {
  try {
    const id = String(req.params.id || "");
    const art = narrationArtifacts.get(id);
    if (!art?.coverPath || !fs.existsSync(art.coverPath)) {
      return res.status(404).json({ error: "Capa não encontrada." });
    }
    const st = await fs.promises.stat(art.coverPath);
    res.setHeader("Content-Type", "image/jpeg");
    res.setHeader("Content-Length", String(st.size));
    fs.createReadStream(art.coverPath).pipe(res);
  } catch (err: any) {
    res.status(500).json({ error: err?.message || "Falha ao servir capa." });
  }
});

// API to stop an active narration stream (true cancel — no partial audio)
app.post("/api/narrate-stop", async (req, res) => {
  const { taskId } = req.body;
  console.log(`[NarrateStop] Request received to stop task: ${taskId}`);
  
  if (taskId && activeTasks.has(taskId)) {
    const task = activeTasks.get(taskId);
    if (task) {
      task.stopped = true;
      // Abort in-flight /tts fetch so the stream exits without encoding
      if (!task.abort.signal.aborted) {
        task.abort.abort();
      }
      console.log(`[NarrateStop] Task ${taskId} marked as stopped and aborted.`);
      // Best-effort interrupt on TTS server (may finish current generate silently).
      // Model unload happens in narrate-stream finally / after cancelled /tts returns.
      void cancelTtsJob(taskId);
      return res.json({ success: true, message: "Narração cancelada." });
    }
  }
  
  res.status(404).json({ error: "Tarefa de narração ativa não encontrada ou já concluída." });
});

// Chunk-cache progress for a document UUID
app.get("/api/chunk-cache/:docId", async (req, res) => {
  const docId = String(req.params.docId || "");
  if (!isSafeDocId(docId)) {
    return res.status(400).json({ error: "docId inválido." });
  }
  const status = await chunkCacheStatus(docId);
  return res.json(status);
});

app.delete("/api/chunk-cache/:docId", async (req, res) => {
  const docId = String(req.params.docId || "");
  if (!isSafeDocId(docId)) {
    return res.status(400).json({ error: "docId inválido." });
  }
  const cleared = await clearChunkCache(docId);
  return res.json({ success: true, cleared });
});

// Document metadata: PDF page count or EPUB chapter outline
app.post("/api/document-info", async (req, res) => {
  let tempPath: string | null = null;
  try {
    const { fileData, pdfData, fileType } = req.body;
    const activeData = fileData || pdfData;
    const type = (fileType || "pdf") as "pdf" | "epub";

    if (!activeData) {
      return res.status(400).json({ error: "O conteúdo do arquivo é obrigatório." });
    }

    if (type === "pdf") {
      const src = await PDFDocument.load(Buffer.from(activeData, "base64"), {
        ignoreEncryption: true,
      });
      const pageCount = src.getPageCount();
      return res.json({
        fileType: "pdf",
        pageCount,
        message: `PDF com ${pageCount} página(s) no arquivo (numeração do leitor PDF, não rótulos impressos).`,
      });
    }

    const tmpDir = os.tmpdir();
    tempPath = path.join(
      tmpDir,
      `temp_epub_info_${Date.now()}_${Math.random().toString(36).substring(7)}.epub`
    );
    await fs.promises.writeFile(tempPath, Buffer.from(activeData, "base64"));
    const chapters = await getEpubOutline(tempPath);
    await fs.promises.unlink(tempPath).catch(() => {});
    tempPath = null;

    return res.json({
      fileType: "epub",
      chapterCount: chapters.length,
      chapters,
      message:
        "EPUB não tem páginas fixas como PDF. Os números abaixo são capítulos/seções da estrutura do livro.",
    });
  } catch (err: any) {
    console.error("[DocumentInfo Error]", err);
    if (tempPath) await fs.promises.unlink(tempPath).catch(() => {});
    res.status(400).json({ error: err.message || "Erro ao ler o documento." });
  }
});

// Extract text only (PDF/EPUB) — used for editable preview before TTS
app.post("/api/extract", async (req, res) => {
  try {
    const { fileData, pdfData, fileType, startPage, endPage } = req.body;
    const activeData = fileData || pdfData;
    const type = (fileType || "pdf") as "pdf" | "epub";

    if (!activeData) {
      return res.status(400).json({ error: "O conteúdo do arquivo é obrigatório." });
    }

    const { start, end } = parsePageRange(startPage, endPage);
    console.log(`[Extract] Type: ${type}, Start: ${start}, End: ${end}`);

    const result = await extractDocumentText({
      fileData: activeData,
      fileType: type,
      start,
      end,
    });

    res.json(result);
  } catch (err: any) {
    console.error("[Extract Error]", err);
    res.status(400).json({ error: err.message || "Erro ao extrair texto do documento." });
  }
});

// PDF/EPUB Extraction and TTS with Stream Progress SSE API
// Accepts either `text` (skip extraction) or fileData for legacy one-shot flow.
app.post("/api/narrate-stream", async (req, res) => {
  // Set headers for SSE stream
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  
  const sendEvent = (data: any) => {
    res.write(`data: ${JSON.stringify(data)}\n\n`);
  };

  let tempPath: string | null = null;
  let taskId: string | undefined = undefined;

  try {
    const {
      pdfData,
      fileData,
      fileType,
      startPage,
      endPage,
      voiceName,
      taskId: reqTaskId,
      text: providedText,
      pagesNarrated: providedPagesLabel,
      outputFormat: reqOutputFormat,
      coverPage: reqCoverPage,
      includeCover: reqIncludeCover,
      sourceFileName,
      docId: reqDocId,
      language: reqLanguage,
    } = req.body;
    taskId = reqTaskId;
    const docId =
      typeof reqDocId === "string" && isSafeDocId(reqDocId) ? reqDocId : null;
    
    if (taskId) {
      activeTasks.set(taskId, { stopped: false, abort: new AbortController() });
    }
    
    const activeData = fileData || pdfData;
    const type = (fileType || "pdf") as "pdf" | "epub";
    const voice = voiceName || "Vivian";
    const engine = activeTtsEngine();
    const languageId = resolveNarrationLanguageIdFromRequest(reqLanguage);
    const ttsLanguage = resolveNarrationLanguagePayload(reqLanguage);
    const outputFormat: "mp3" | "m4b" =
      reqOutputFormat === "m4b" ? "m4b" : "mp3";
    const includeCover = reqIncludeCover !== false;
    const coverPageNum =
      reqCoverPage === "" || reqCoverPage == null || reqCoverPage === undefined
        ? null
        : Math.max(1, parseInt(String(reqCoverPage), 10) || 1);

    let start = 1;
    let end = 1;
    try {
      ({ start, end } = parsePageRange(startPage, endPage));
    } catch (rangeErr: any) {
      if (!providedText) {
        sendEvent({ type: "error", error: rangeErr.message });
        return res.end();
      }
    }

    console.log(
      `[NarrateStream] textProvided=${!!providedText}, Type: ${type}, Start: ${start}, End: ${end}, Voice: ${voice}, Language: ${ttsLanguage}`
    );

    let extractedText = "";
    let pagesLabel =
      providedPagesLabel ||
      (type === "pdf"
        ? start === end
          ? `Página ${start}`
          : `Páginas ${start} - ${end}`
        : start === end
          ? `Capítulo/Seção ${start}`
          : `Capítulos/Seções ${start} - ${end}`);

    if (typeof providedText === "string" && providedText.trim()) {
      sendEvent({
        type: "status",
        step: "pre_tts",
        message: "Preparando texto editado para narração...",
      });
      extractedText = textForEngine(
        sanitizeExtractedText(providedText, engine),
        engine
      );
      if (!extractedText) {
        sendEvent({ type: "error", error: "O texto para narração está vazio." });
        return res.end();
      }
      sendEvent({
        type: "status",
        step: "pre_tts",
        message: "Texto pronto. Dividindo o conteúdo para narração...",
        extractedText,
      });
    } else {
      if (!activeData) {
        sendEvent({ type: "error", error: "O conteúdo do arquivo é obrigatório." });
        return res.end();
      }

      sendEvent({
        type: "status",
        step: "init",
        message:
          type === "pdf"
            ? "Iniciando leitura e preparando o PDF..."
            : "Iniciando leitura e preparando o EPUB...",
      });

      sendEvent({
        type: "status",
        step: "extraction",
        message:
          type === "pdf"
            ? `Extraindo texto das páginas ${start}${start === end ? "" : ` a ${end}`}...`
            : `Extraindo texto do EPUB a partir da seção/capítulo ${start}...`,
      });

      try {
        const result = await extractDocumentText({
          fileData: activeData,
          fileType: type,
          start,
          end,
        });
        extractedText = result.extractedText;
        pagesLabel = result.pagesNarrated;
      } catch (extractErr: any) {
        sendEvent({ type: "error", error: extractErr.message || "Falha na extração." });
        return res.end();
      }

      sendEvent({
        type: "status",
        step: "pre_tts",
        message: "Texto extraído com sucesso! Dividindo o conteúdo para narração...",
        extractedText,
      });
    }

    const textChunks = splitTextIntoChunks(extractedText, engine);
    const totalChunks = textChunks.length;

    if (totalChunks === 0) {
      sendEvent({ type: "error", error: "O texto extraído do documento está vazio." });
      return res.end();
    }

    sendEvent({ 
      type: "status", 
      step: "chunks", 
      message: `Conteúdo dividido em ${totalChunks} partes para processamento.`,
      current: 0,
      total: totalChunks,
    });

    // Qwen: ensure preview WAV+TXT for ICL. Kokoro: speaker id only.
    let voiceAnchor: { refAudioPath: string; refText: string } | null = null;
    if (engine === "qwen3") {
      sendEvent({
        type: "status",
        step: "pre_tts",
        message: "Preparando âncora de voz (prévia) para tom consistente...",
      });
      try {
        voiceAnchor = await ensureVoicePreview(voice, languageId);
        console.log(`[NarrateStream] Voice anchor: ${voiceAnchor.refAudioPath}`);
      } catch (anchorErr: any) {
        sendEvent({
          type: "error",
          error:
            anchorErr?.message ||
            "Falha ao preparar a âncora de voz. Gere a prévia da voz no seletor e tente de novo.",
        });
        return res.end();
      }
    }

    // Run Text-To-Speech via local engine for each chunk — PCM cached per doc UUID
    ensureNarrationTmpDir();
    const audioId = newNarrationId();
    const pcmPath = path.join(NARRATION_TMP_DIR, `${audioId}.pcm`);
    const mp3Path = path.join(NARRATION_TMP_DIR, `${audioId}.mp3`);
    let sampleRate = 24000;
    let wasStoppedEarly = false;
    const taskAbort = taskId ? activeTasks.get(taskId)?.abort : undefined;
    const engineLabel = engine === "kokoro" ? "Kokoro" : "Qwen3";
    const fingerprint = narrationFingerprint(
      extractedText,
      voice,
      engine,
      ttsLanguage
    );

    let cacheMeta: ChunkCacheMeta | null = null;
    if (docId) {
      cacheMeta = await ensureChunkCache(
        docId,
        voice,
        engine,
        fingerprint,
        totalChunks,
        sampleRate
      );
      if (cacheMeta.completedIndices.length > 0) {
        sendEvent({
          type: "status",
          step: "chunks",
          message: `Retomando cache: ${cacheMeta.completedIndices.length} de ${totalChunks} blocos já narrados.`,
          current: cacheMeta.completedIndices.length,
          total: totalChunks,
          cached: cacheMeta.completedIndices.length,
        });
      }
    }

    const completedSet = new Set(cacheMeta?.completedIndices ?? []);

    for (let i = 0; i < totalChunks; i++) {
      // Check if task has been cancelled / stopped
      if (taskId) {
        const task = activeTasks.get(taskId);
        if (task?.stopped || task?.abort.signal.aborted) {
          console.log(`[NarrateStream] Task ${taskId} was stopped by user at chunk index ${i}.`);
          wasStoppedEarly = true;
          break;
        }
      }

      const chunk = textChunks[i];
      const partNum = i + 1;

      if (completedSet.has(i) && docId) {
        sendEvent({
          type: "status",
          step: "tts",
          current: partNum,
          total: totalChunks,
          cached: true,
          message: `Reutilizando bloco ${partNum} de ${totalChunks} (cache)...`,
        });
        continue;
      }

      sendEvent({
        type: "status",
        step: "tts",
        current: partNum,
        total: totalChunks,
        message: `Narrando parte ${partNum} de ${totalChunks} (${engineLabel})...`,
      });

      const breakSeconds =
        parseBreakSeconds(chunk) ?? (chunk.trim() === "..." ? 0.25 : null);

      let pcm: Buffer = Buffer.alloc(0);
      let cancelled = false;

      if (breakSeconds != null) {
        pcm = silencePcmS16le(sampleRate, breakSeconds);
      } else {
        const heartbeat = setInterval(() => {
          sendEvent({
            type: "status",
            step: "tts",
            current: partNum,
            total: totalChunks,
            message: `Narrando parte ${partNum} de ${totalChunks} — ${engineLabel} ainda gerando...`,
          });
        }, 10_000);

        try {
          const result = await synthesizeWithTts(
            chunk,
            voice,
            taskId,
            taskAbort?.signal,
            voiceAnchor
              ? {
                  refAudioPath: voiceAnchor.refAudioPath,
                  refText: voiceAnchor.refText,
                  language: ttsLanguage,
                  instruct: mergeQwenInstruct(languageId, TTS_INSTRUCT),
                }
              : {
                  skipIcl: true,
                  language: ttsLanguage,
                  instruct: mergeQwenInstruct(languageId, TTS_INSTRUCT),
                }
          );
          pcm = result.pcm;
          sampleRate = result.sampleRate;
          cancelled = result.cancelled;
          if (i === 0) {
            console.log(
              `[NarrateStream] Chunk 1 TTS done (engine=${engine}, voice=${voice}, icl=${!!result.icl}, cancelled=${cancelled})`
            );
          }
          if (engine === "qwen3" && !result.icl && !cancelled) {
            console.warn(
              `[NarrateStream] Chunk ${partNum}: ICL not used for voice=${voice} — ` +
                "preview anchor may be missing or Base encoder not ready."
            );
          }
        } catch (ttsErr: any) {
          console.warn(`[NarrateStream] Chunk ${partNum} failed:`, ttsErr?.message || ttsErr);
          if (taskId && activeTasks.get(taskId)?.stopped) {
            wasStoppedEarly = true;
            break;
          }
          continue;
        } finally {
          clearInterval(heartbeat);
        }
      }

      if (cancelled) {
        console.log(`[NarrateStream] TTS cancelled mid-chunk at index ${i}.`);
        wasStoppedEarly = true;
        break;
      }

      if (pcm.length > 0 && docId && cacheMeta) {
        cacheMeta = await saveChunkPcm(docId, i, pcm, {
          ...cacheMeta,
          sampleRate,
        });
        completedSet.add(i);
      } else if (pcm.length > 0 && !docId) {
        // Fallback without doc UUID: append to ephemeral PCM (no resume)
        await fs.promises.appendFile(pcmPath, pcm);
        completedSet.add(i);
      }
    }

    const completedCount = completedSet.size;

    // True cancel: keep disk chunk cache for resume, do not encode partial audio
    if (wasStoppedEarly) {
      await fs.promises.unlink(pcmPath).catch(() => undefined);
      sendEvent({
        type: "cancelled",
        docId,
        completed: completedCount,
        total: totalChunks,
        message: `Narração cancelada. ${completedCount} de ${totalChunks} blocos salvos para retomar.`,
      });
      return res.end();
    }

    if (completedCount === 0) {
      await fs.promises.unlink(pcmPath).catch(() => undefined);
      sendEvent({
        type: "error",
        error: `A geração de áudio falhou para todas as partes do texto. Verifique se o servidor ${engineLabel} TTS está rodando.`,
      });
      return res.end();
    }

    if (completedCount < totalChunks) {
      await fs.promises.unlink(pcmPath).catch(() => undefined);
      sendEvent({
        type: "error",
        error: `Narração incompleta: ${completedCount} de ${totalChunks} blocos. Tente novamente para retomar o cache.`,
      });
      return res.end();
    }

    sendEvent({
      type: "status",
      step: "encoding",
      current: completedCount,
      total: totalChunks,
      percent: 0,
      message:
        outputFormat === "m4b"
          ? "Codificando áudio e montando o arquivo M4B..."
          : "Codificando e compactando áudio para o formato MP3...",
    });

    let pcmBytes = 0;
    let pcmParts = completedCount;
    if (docId) {
      const concat = await concatCachedChunksToPcm(docId, totalChunks, pcmPath);
      pcmBytes = concat.bytes;
      pcmParts = concat.parts;
      if (cacheMeta) {
        cacheMeta.sampleRate = sampleRate;
        await writeChunkCacheMeta(cacheMeta);
      }
    } else {
      try {
        const st = await fs.promises.stat(pcmPath);
        pcmBytes = st.size;
      } catch {
        pcmBytes = 0;
      }
    }

    if (pcmBytes === 0) {
      await fs.promises.unlink(pcmPath).catch(() => undefined);
      sendEvent({
        type: "error",
        error: "Nenhum áudio PCM disponível para codificar.",
      });
      return res.end();
    }

    const bookStem = sanitizeDownloadBaseName(
      String(sourceFileName || "")
        .replace(/\.[^/.]+$/, "")
        .trim() || "narracao"
    );
    const downloadBase = bookStem;

    // Extract cover JPEG when requested
    let coverPath: string | null = null;
    if (includeCover && activeData) {
      try {
        if (type === "pdf" && coverPageNum != null) {
          const cover = await extractCover({
            fileData: activeData,
            fileType: "pdf",
            coverPage: coverPageNum,
          });
          coverPath = cover.jpegPath;
        } else if (type === "epub") {
          const bytes = Buffer.from(activeData, "base64");
          const { coverJpegPath } = await extractEpubImages(bytes);
          coverPath = coverJpegPath;
        }
      } catch (coverErr: any) {
        console.warn(`[NarrateStream] Cover extraction skipped:`, coverErr?.message || coverErr);
      }
    } else if (includeCover && type === "pdf" && coverPageNum != null && !activeData) {
      // Text-only narrate without file — cannot extract cover
      console.warn(`[NarrateStream] Cover requested but no fileData provided.`);
    }

    let finalAudioPath = mp3Path;
    let finalFormat: "mp3" | "m4b" = "mp3";
    let mimeType = "audio/mpeg";

    try {
      await ensureFfmpeg();
      if (outputFormat === "m4b") {
        const m4bPath = path.join(NARRATION_TMP_DIR, `${audioId}.m4b`);
        await pcmToM4b({
          pcmPath,
          outputPath: m4bPath,
          sampleRate,
          artworkPaths: coverPath ? [coverPath] : [],
          title: bookStem,
          onProgress: (p) => {
            const pct = p.percent != null ? Math.round(p.percent) : null;
            sendEvent({
              type: "status",
              step: "encoding",
              current: pcmParts,
              total: totalChunks,
              percent: pct,
              message:
                p.percent != null
                  ? `Normalizando e codificando M4B… ${Math.round(p.percent)}%`
                  : "Normalizando e codificando M4B…",
            });
          },
          signal: taskAbort?.signal,
        });
        finalAudioPath = m4bPath;
        finalFormat = "m4b";
        mimeType = "audio/mp4";
      } else {
        await pcmToMp3({
          pcmPath,
          outputPath: mp3Path,
          sampleRate,
          onProgress: (p) => {
            const pct = p.percent != null ? Math.round(p.percent) : null;
            sendEvent({
              type: "status",
              step: "encoding",
              current: pcmParts,
              total: totalChunks,
              percent: pct,
              message:
                p.percent != null
                  ? `Normalizando e codificando MP3… ${Math.round(p.percent)}%`
                  : "Normalizando e codificando MP3…",
            });
          },
          signal: taskAbort?.signal,
        });
      }
    } catch (encodeErr: any) {
      sendEvent({
        type: "error",
        error:
          encodeErr?.message ||
          `Falha ao gerar o arquivo ${outputFormat === "m4b" ? "M4B" : "MP3"}.`,
      });
      return res.end();
    }
    const encodedBytes = (await fs.promises.stat(finalAudioPath)).size;
    console.log(
      `[NarrateStream] Encoded ${finalFormat.toUpperCase()} ${encodedBytes} bytes from ${pcmBytes} PCM bytes (${pcmParts} parts)`
    );

    narrationArtifacts.set(audioId, {
      mp3Path: finalAudioPath,
      pcmPath,
      fileName: downloadBase,
      createdAt: Date.now(),
      format: finalFormat,
      coverPath,
      mimeType,
    });
    console.log(
      `[NarrateStream] Áudio salvo: nome="${path.basename(finalAudioPath)}" caminho="${path.resolve(finalAudioPath)}"`
    );
    // PCM no longer needed after encode
    await fs.promises.unlink(pcmPath).catch(() => undefined);
    // Chunk cache was consumed into the final audio — clear after use
    if (docId) {
      await clearChunkCache(docId);
    }

    sendEvent({
      type: "done",
      audioId,
      audioUrl: `/api/narration-audio/${audioId}`,
      coverUrl: coverPath ? `/api/narration-cover/${audioId}` : null,
      format: finalFormat,
      voiceName: voice,
      pagesNarrated: pagesLabel,
      docId,
    });

    res.end();

  } catch (err: any) {
    console.error("[NarrateStream Server Error]", err);
    if (tempPath) {
      await fs.promises.unlink(tempPath).catch(() => {});
    }
    try {
      sendEvent({ type: "error", error: err.message || "Ocorreu um erro interno ao processar o áudio." });
    } catch {
      // response may already be broken (e.g. OOM)
    }
    res.end();
  } finally {
    await unloadTtsModel();
    if (taskId) {
      activeTasks.delete(taskId);
      console.log(`[NarrateStream] Cleaned up taskId ${taskId} from activeTasks.`);
    }
  }
});

// Original endpoint kept for backwards compatibility / fallback
app.post("/api/narrate", async (req, res) => {
  try {
    const { pdfData, startPage, endPage, voiceName, language: reqLanguage } = req.body;

    if (!pdfData) {
      return res.status(400).json({ error: "O arquivo PDF é obrigatório." });
    }

    const { start, end } = parsePageRange(startPage, endPage);
    const voice = voiceName || "Vivian";
    const languageId = resolveNarrationLanguageIdFromRequest(reqLanguage);
    const ttsLanguage = resolveNarrationLanguagePayload(reqLanguage);

    console.log(
      `[Narrate Legacy] Start Page: ${start}, End Page: ${end}, Voice: ${voice}, Language: ${ttsLanguage}`
    );

    const { extractedText, pagesNarrated } = await extractDocumentText({
      fileData: pdfData,
      fileType: "pdf",
      start,
      end,
    });

    const engine = activeTtsEngine();
    const textChunks = splitTextIntoChunks(extractedText, engine);
    let voiceAnchor: { refAudioPath: string; refText: string } | null = null;
    if (engine === "qwen3") {
      voiceAnchor = await ensureVoicePreview(voice, languageId);
    }

    ensureNarrationTmpDir();
    const audioId = newNarrationId();
    const pcmPath = path.join(NARRATION_TMP_DIR, `${audioId}.pcm`);
    const mp3Path = path.join(NARRATION_TMP_DIR, `${audioId}.mp3`);
    let pcmBytes = 0;
    let sampleRate = 24000;
    let failedChunk: number | null = null;

    for (let i = 0; i < textChunks.length; i++) {
      const chunk = textChunks[i];
      try {
        const breakSeconds =
          parseBreakSeconds(chunk) ?? (chunk.trim() === "..." ? 0.25 : null);
        if (breakSeconds != null) {
          const pcm = silencePcmS16le(sampleRate, breakSeconds);
          if (pcm.length > 0) {
            await fs.promises.appendFile(pcmPath, pcm);
            pcmBytes += pcm.length;
          }
          continue;
        }

        const { pcm, sampleRate: sr } = await synthesizeWithTts(
          chunk,
          voice,
          undefined,
          undefined,
          voiceAnchor
            ? {
                refAudioPath: voiceAnchor.refAudioPath,
                refText: voiceAnchor.refText,
                language: ttsLanguage,
                instruct: mergeQwenInstruct(languageId, TTS_INSTRUCT),
              }
            : {
                skipIcl: true,
                language: ttsLanguage,
                instruct: mergeQwenInstruct(languageId, TTS_INSTRUCT),
              }
        );
        sampleRate = sr;
        if (pcm.length > 0) {
          await fs.promises.appendFile(pcmPath, pcm);
          pcmBytes += pcm.length;
        }
      } catch (ttsErr: any) {
        console.warn(`[Narrate Legacy] Chunk ${i + 1} failed:`, ttsErr?.message || ttsErr);
        failedChunk = i + 1;
        break;
      }
    }

    if (failedChunk != null) {
      await fs.promises.unlink(pcmPath).catch(() => undefined);
      return res.status(500).json({
        error: `Narração incompleta: falha no bloco ${failedChunk} de ${textChunks.length}.`,
      });
    }

    if (pcmBytes === 0) {
      await fs.promises.unlink(pcmPath).catch(() => undefined);
      return res.status(500).json({
        error: `A geração de áudio falhou para todas as partes do texto. Verifique se o servidor ${engine === "kokoro" ? "Kokoro" : "Qwen3"} TTS está rodando.`,
      });
    }

    await ensureFfmpeg();
    await pcmToMp3({ pcmPath, outputPath: mp3Path, sampleRate });
    await fs.promises.unlink(pcmPath).catch(() => undefined);
    narrationArtifacts.set(audioId, {
      mp3Path,
      pcmPath,
      fileName: sanitizeDownloadBaseName(pagesNarrated),
      createdAt: Date.now(),
    });
    console.log(
      `[NarrateLegacy] Áudio salvo: nome="${path.basename(mp3Path)}" caminho="${path.resolve(mp3Path)}"`
    );

    res.json({
      audioId,
      audioUrl: `/api/narration-audio/${audioId}`,
      extractedText: extractedText,
      voiceName: voice,
      pagesNarrated
    });

  } catch (error: any) {
    console.error("[Narrate Legacy Error]", error);
    res.status(500).json({ error: error.message || "Erro interno ao processar e narrar o PDF." });
  } finally {
    await unloadTtsModel();
  }
});

// --- Cover preview / extract ---
app.post("/api/cover-preview", async (req, res) => {
  try {
    const { fileData, fileType, coverPage } = req.body as {
      fileData?: string;
      fileType?: "pdf" | "epub";
      coverPage?: number | string | null;
    };
    if (!fileData) {
      return res.status(400).json({ error: "O conteúdo do arquivo é obrigatório." });
    }
    const type = fileType === "epub" ? "epub" : "pdf";
    const page =
      coverPage === "" || coverPage == null
        ? type === "pdf"
          ? 1
          : null
        : Math.max(1, parseInt(String(coverPage), 10) || 1);

    if (type === "pdf" && page == null) {
      return res.json({ found: false, message: "Informe a página da capa." });
    }

    const cover = await extractCover({
      fileData,
      fileType: type,
      coverPage: page,
    });
    const encoded = await coverToBase64Jpeg(cover.jpegPath);
    // Keep tmp for a bit; preview is ephemeral — unlink after encoding
    await fs.promises.unlink(cover.jpegPath).catch(() => undefined);

    return res.json({
      found: true,
      imageData: encoded.imageData,
      mimeType: encoded.mimeType,
      width: encoded.width || cover.width,
      height: encoded.height || cover.height,
      source: cover.source,
    });
  } catch (err: any) {
    console.error("[CoverPreview]", err);
    return res.status(400).json({
      found: false,
      error: err?.message || "Não foi possível gerar o preview da capa.",
    });
  }
});

app.post("/api/chapter-preview", async (req, res) => {
  try {
    const { fileData, chapterIndex } = req.body as {
      fileData?: string;
      chapterIndex?: number | string | null;
    };
    if (!fileData) {
      return res.status(400).json({ error: "O conteúdo do arquivo é obrigatório." });
    }
    const index = Math.max(1, parseInt(String(chapterIndex ?? 1), 10) || 1);
    const bytes = Buffer.from(fileData, "base64");
    const preview = await getEpubChapterPreview(bytes, index);
    return res.json({ found: true, ...preview });
  } catch (err: any) {
    console.error("[ChapterPreview]", err);
    return res.status(400).json({
      found: false,
      error: err?.message || "Não foi possível gerar o preview do capítulo.",
    });
  }
});

app.post("/api/extract-cover", async (req, res) => {
  try {
    const { fileData, fileType, coverPage, fileName } = req.body as {
      fileData?: string;
      fileType?: "pdf" | "epub";
      coverPage?: number | string | null;
      fileName?: string;
    };
    if (!fileData) {
      return res.status(400).json({ error: "O conteúdo do arquivo é obrigatório." });
    }
    const type = fileType === "epub" ? "epub" : "pdf";
    const page =
      coverPage === "" || coverPage == null
        ? type === "pdf"
          ? 1
          : null
        : Math.max(1, parseInt(String(coverPage), 10) || 1);

    const cover = await extractCover({
      fileData,
      fileType: type,
      coverPage: page,
    });

    const stem = sanitizeDownloadBaseName(
      String(fileName || "capa")
        .replace(/\.[^/.]+$/, "")
        .trim() || "capa"
    );
    const saved = await copyToDownloads(cover.jpegPath, `${stem}.jpg`);
    await fs.promises.unlink(cover.jpegPath).catch(() => undefined);

    return res.json({
      success: true,
      path: saved.path,
      fileName: saved.fileName,
      directory: saved.directory,
    });
  } catch (err: any) {
    console.error("[ExtractCover]", err);
    return res.status(400).json({
      error: err?.message || "Não foi possível extrair a capa.",
    });
  }
});

app.post(
  "/api/convert/mp3-to-m4b",
  convertUpload.fields([
    { name: "mp3", maxCount: 1 },
    { name: "cover", maxCount: 1 },
  ]),
  async (req, res) => {
    let tmpCover: string | null = null;
    let tmpRawImage: string | null = null;
    let tmpM4b: string | null = null;
    const uploaded: string[] = [];
    let sseStarted = false;
    const abort = new AbortController();
    const onClientGone = () => {
      if (!res.writableEnded && !abort.signal.aborted) abort.abort();
    };
    req.on("aborted", onClientGone);
    res.on("close", onClientGone);

    const startSse = () => {
      if (sseStarted) return;
      sseStarted = true;
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");
      (res as any).flushHeaders?.();
    };

    const sendEvent = (data: Record<string, unknown>) => {
      if (abort.signal.aborted) return;
      startSse();
      try {
        res.write(`data: ${JSON.stringify(data)}\n\n`);
      } catch {
        // connection closed
      }
    };

    try {
      await ensureFfmpeg();
      const files = req.files as { [field: string]: Express.Multer.File[] } | undefined;
      const mp3File = files?.mp3?.[0];
      const coverFile = files?.cover?.[0];
      const fileType = String(req.body?.fileType || "image") as "pdf" | "epub" | "image";
      const coverPage = req.body?.coverPage as string | undefined;
      const fileName = String(req.body?.fileName || mp3File?.originalname || "audio");

      if (!mp3File?.path) {
        return res.status(400).json({ error: "O arquivo MP3 é obrigatório." });
      }
      if (!coverFile?.path) {
        return res.status(400).json({
          error: "A capa é obrigatória (imagem JPEG/PNG/WebP/GIF, PDF ou EPUB).",
        });
      }
      uploaded.push(mp3File.path, coverFile.path);

      sendEvent({
        type: "status",
        stage: "cover",
        percent: 0,
        message: "Preparando capa...",
      });

      if (fileType === "image") {
        const ext = path.extname(coverFile.originalname || "").replace(/^\./, "").toLowerCase() || "jpg";
        const safeExt = ["jpg", "jpeg", "png", "webp", "gif", "bmp", "tiff", "tif"].includes(ext)
          ? ext.replace(/jpeg/, "jpg")
          : "jpg";
        tmpRawImage = `${coverFile.path}.${safeExt}`;
        await fs.promises.rename(coverFile.path, tmpRawImage);
        uploaded[1] = tmpRawImage;
        tmpCover = await convertImageToJpeg(tmpRawImage);
      } else {
        const type = fileType === "epub" ? "epub" : "pdf";
        const page =
          coverPage === "" || coverPage == null
            ? type === "pdf"
              ? 1
              : null
            : Math.max(1, parseInt(String(coverPage), 10) || 1);
        const buf = await fs.promises.readFile(coverFile.path);
        const cover = await extractCover({
          fileData: buf.toString("base64"),
          fileType: type,
          coverPage: page,
        });
        tmpCover = cover.jpegPath;
      }

      const stem = sanitizeDownloadBaseName(
        String(fileName).replace(/\.[^/.]+$/, "").trim() || "audio"
      );
      tmpM4b = path.join(NARRATION_TMP_DIR, `convert-${Date.now().toString(36)}.m4b`);
      ensureNarrationTmpDir();

      sendEvent({
        type: "status",
        stage: "encode",
        percent: 0,
        message: "Convertendo áudio para M4B (AAC)...",
      });

      await mp3ToM4b({
        mp3Path: mp3File.path,
        outputPath: tmpM4b,
        artworkPaths: tmpCover ? [tmpCover] : [],
        title: stem,
        signal: abort.signal,
        onProgress: (p) => {
          sendEvent({
            type: "status",
            stage: "encode",
            percent: p.percent != null ? Math.round(p.percent) : null,
            timeSec: Math.round(p.timeSec),
            totalSec: p.totalSec != null ? Math.round(p.totalSec) : null,
            message:
              p.percent != null
                ? `Convertendo… ${Math.round(p.percent)}%`
                : `Convertendo… ${Math.round(p.timeSec)}s processados`,
          });
        },
      });

      if (abort.signal.aborted) {
        sendEvent({ type: "error", error: "Conversão cancelada." });
        res.end();
        return;
      }

      sendEvent({
        type: "status",
        stage: "save",
        percent: 100,
        message: "Salvando em Downloads...",
      });

      const savedAudio = await copyToDownloads(tmpM4b, `${stem}.m4b`);
      const savedCover = tmpCover
        ? await copyToDownloads(tmpCover, `${stem}.jpg`)
        : null;

      sendEvent({
        type: "done",
        success: true,
        fileName: savedAudio.fileName,
        path: savedAudio.path,
        coverFileName: savedCover?.fileName ?? null,
        coverPath: savedCover?.path ?? null,
        directory: savedAudio.directory,
      });
      res.end();
    } catch (err: any) {
      console.error("[ConvertMp3ToM4b]", err);
      const cancelled =
        abort.signal.aborted ||
        err?.name === "AbortError" ||
        /cancelad/i.test(String(err?.message || ""));
      if (sseStarted) {
        sendEvent({
          type: "error",
          error: cancelled
            ? "Conversão cancelada."
            : err?.message || "Falha ao converter MP3 para M4B.",
        });
        res.end();
      } else {
        return res.status(400).json({
          error: cancelled
            ? "Conversão cancelada."
            : err?.message || "Falha ao converter MP3 para M4B.",
        });
      }
    } finally {
      req.off("aborted", onClientGone);
      res.off("close", onClientGone);
      for (const p of uploaded) await fs.promises.unlink(p).catch(() => undefined);
      if (tmpRawImage && !uploaded.includes(tmpRawImage)) {
        await fs.promises.unlink(tmpRawImage).catch(() => undefined);
      }
      if (tmpCover) await fs.promises.unlink(tmpCover).catch(() => undefined);
      if (tmpM4b) await fs.promises.unlink(tmpM4b).catch(() => undefined);
    }
  }
);

app.post(
  "/api/convert/m4b-to-mp3",
  convertUpload.single("m4b"),
  async (req, res) => {
    let tmpMp3: string | null = null;
    let tmpCover: string | null = null;
    const uploadedPath = req.file?.path || null;
    let sseStarted = false;
    const abort = new AbortController();
    const onClientGone = () => {
      if (!res.writableEnded && !abort.signal.aborted) abort.abort();
    };
    req.on("aborted", onClientGone);
    res.on("close", onClientGone);

    const startSse = () => {
      if (sseStarted) return;
      sseStarted = true;
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");
      (res as any).flushHeaders?.();
    };

    const sendEvent = (data: Record<string, unknown>) => {
      if (abort.signal.aborted) return;
      startSse();
      try {
        res.write(`data: ${JSON.stringify(data)}\n\n`);
      } catch {
        // closed
      }
    };

    try {
      await ensureFfmpeg();
      if (!uploadedPath) {
        return res.status(400).json({ error: "O arquivo M4B é obrigatório." });
      }

      const fileName = String(req.body?.fileName || req.file?.originalname || "audio");

      sendEvent({
        type: "status",
        stage: "encode",
        percent: 0,
        message: "Extraindo áudio MP3...",
      });

      const result = await m4bToMp3AndCover({
        m4bPath: uploadedPath,
        signal: abort.signal,
        onProgress: (p) => {
          sendEvent({
            type: "status",
            stage: "encode",
            percent: p.percent != null ? Math.round(p.percent) : null,
            timeSec: Math.round(p.timeSec),
            totalSec: p.totalSec != null ? Math.round(p.totalSec) : null,
            message:
              p.percent != null
                ? `Extraindo… ${Math.round(p.percent)}%`
                : `Extraindo… ${Math.round(p.timeSec)}s processados`,
          });
        },
      });
      tmpMp3 = result.mp3Path;
      tmpCover = result.coverPath;

      if (abort.signal.aborted) {
        sendEvent({ type: "error", error: "Conversão cancelada." });
        res.end();
        return;
      }

      sendEvent({
        type: "status",
        stage: "save",
        percent: 100,
        message: "Salvando em Downloads...",
      });

      const stem = sanitizeDownloadBaseName(
        String(fileName).replace(/\.[^/.]+$/, "").trim() || "audio"
      );
      const savedAudio = await copyToDownloads(tmpMp3, `${stem}.mp3`);
      let coverFileName: string | null = null;
      let coverPathOut: string | null = null;
      if (tmpCover) {
        const savedCover = await copyToDownloads(tmpCover, `${stem}.jpg`);
        coverFileName = savedCover.fileName;
        coverPathOut = savedCover.path;
      }

      sendEvent({
        type: "done",
        success: true,
        fileName: savedAudio.fileName,
        path: savedAudio.path,
        coverFileName,
        coverPath: coverPathOut,
        directory: savedAudio.directory,
        hasCover: !!tmpCover,
      });
      res.end();
    } catch (err: any) {
      console.error("[ConvertM4bToMp3]", err);
      const cancelled =
        abort.signal.aborted ||
        err?.name === "AbortError" ||
        /cancelad/i.test(String(err?.message || ""));
      if (sseStarted) {
        sendEvent({
          type: "error",
          error: cancelled
            ? "Conversão cancelada."
            : err?.message || "Falha ao converter M4B para MP3.",
        });
        res.end();
      } else {
        return res.status(400).json({
          error: cancelled
            ? "Conversão cancelada."
            : err?.message || "Falha ao converter M4B para MP3.",
        });
      }
    } finally {
      req.off("aborted", onClientGone);
      res.off("close", onClientGone);
      if (uploadedPath) await fs.promises.unlink(uploadedPath).catch(() => undefined);
      if (tmpMp3) await fs.promises.unlink(tmpMp3).catch(() => undefined);
      if (tmpCover) await fs.promises.unlink(tmpCover).catch(() => undefined);
    }
  }
);

// Express error handler (payload / multer)
app.use((err: any, _req: express.Request, res: express.Response, next: express.NextFunction) => {
  if (err?.type === "entity.too.large" || err?.name === "PayloadTooLargeError") {
    return res.status(413).json({
      error: "Arquivo grande demais para este tipo de envio.",
    });
  }
  if (err instanceof multer.MulterError) {
    return res.status(400).json({
      error:
        err.code === "LIMIT_FILE_SIZE"
          ? "Arquivo excede o limite de 2 GB."
          : err.message || "Falha no upload.",
    });
  }
  return next(err);
});

// Configure Vite middleware in development or serve static built files in production
async function start() {
  if (process.env.NODE_ENV !== "production") {
    const { createServer: createViteServer } = await import("vite");
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(AURA_ROOT, "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://0.0.0.0:${PORT}`);
  });
}

start();
