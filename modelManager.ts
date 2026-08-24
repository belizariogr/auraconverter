/**
 * Local TTS model install status + downloads (no Python).
 * Supports Qwen3 (HF repos by platform) and Kokoro (MLX on macOS, ONNX elsewhere).
 */
import fs from "fs";
import path from "path";
import { pipeline } from "stream/promises";
import { Readable } from "stream";
import type { ReadableStream as WebReadableStream } from "stream/web";
import { detectGpu, type GpuDetectResult } from "./gpuDetect";
import { probeKokoroAccel } from "./kokoroAccel";
import {
  defaultKokoroBackend,
  readKokoroBackend,
  readTtsEngine,
  readKokoroDevice,
  type KokoroBackendId,
  type TtsEngineId,
  voicesForEngine,
} from "./ttsEngine";
import {
  cancelRuntimeSetup,
  ensureEngineRuntime,
  isEngineRuntimeReady,
} from "./ttsRuntime";

export type ModelSpec = {
  id: string;
  folder: string;
  label: string;
  /** Hugging Face repo (Qwen) or empty for direct URL assets (Kokoro). */
  repo: string;
  approxBytes: number;
  /** Direct file downloads (Kokoro). */
  files?: Array<{ name: string; url: string; approxBytes: number }>;
};

const MLX_MODELS: ModelSpec[] = [
  {
    id: "base",
    folder: "Qwen3-TTS-12Hz-1.7B-Base-8bit",
    label: "Base 1.7B (ICL / clonagem de voz)",
    repo: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
    approxBytes: 2_800_000_000,
  },
  {
    id: "custom",
    folder: "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
    label: "CustomVoice 1.7B (prévias e speakers)",
    repo: "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
    approxBytes: 2_800_000_000,
  },
];

const TORCH_MODELS: ModelSpec[] = [
  {
    id: "base",
    folder: "Qwen3-TTS-12Hz-1.7B-Base",
    label: "Base 1.7B (ICL / clonagem de voz)",
    repo: "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    approxBytes: 3_500_000_000,
  },
  {
    id: "custom",
    folder: "Qwen3-TTS-12Hz-1.7B-CustomVoice",
    label: "CustomVoice 1.7B (prévias e speakers)",
    repo: "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    approxBytes: 3_500_000_000,
  },
];

const KOKORO_RELEASE =
  "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0";

const KOKORO_ONNX_MODELS: ModelSpec[] = [
  {
    id: "kokoro",
    folder: "kokoro",
    label: "Kokoro 82M (ONNX + vozes)",
    repo: "",
    approxBytes: 325_000_000 + 28_000_000,
    files: [
      {
        name: "kokoro-v1.0.onnx",
        url: `${KOKORO_RELEASE}/kokoro-v1.0.onnx`,
        approxBytes: 325_000_000,
      },
      {
        name: "voices-v1.0.bin",
        url: `${KOKORO_RELEASE}/voices-v1.0.bin`,
        approxBytes: 28_000_000,
      },
    ],
  },
];

/** Full-quality bf16 (not quantized) — same audio quality as the ONNX fp-ish release. */
const KOKORO_MLX_MODELS: ModelSpec[] = [
  {
    id: "kokoro",
    folder: "Kokoro-82M-bf16",
    label: "Kokoro 82M (MLX bf16)",
    repo: "mlx-community/Kokoro-82M-bf16",
    approxBytes: 327_000_000,
  },
];

export function isTorchTtsPlatform(platform = process.platform): boolean {
  return platform === "win32" || platform === "linux";
}

/** Kokoro uses MLX on Apple Silicon; ONNX on Windows/Linux. */
export function isKokoroMlxPlatform(platform = process.platform): boolean {
  return platform === "darwin";
}

export function getQwenModels(platform = process.platform): ModelSpec[] {
  return isTorchTtsPlatform(platform) ? TORCH_MODELS : MLX_MODELS;
}

export function getKokoroModels(
  platform = process.platform,
  backend: KokoroBackendId = defaultKokoroBackend(platform)
): ModelSpec[] {
  return backend === "mlx" && isKokoroMlxPlatform(platform)
    ? KOKORO_MLX_MODELS
    : KOKORO_ONNX_MODELS;
}

export function getRequiredModels(
  engine: TtsEngineId = "qwen3",
  platform = process.platform,
  kokoroBackend: KokoroBackendId = defaultKokoroBackend(platform)
): ModelSpec[] {
  if (engine === "kokoro") return getKokoroModels(platform, kokoroBackend);
  return getQwenModels(platform);
}

/** Local folder for one Kokoro/Qwen model spec under the engine models root. */
export function modelSpecLocalDir(modelsDir: string, spec: ModelSpec): string {
  if (spec.files?.length) return modelsDir;
  return path.join(modelsDir, spec.folder);
}

/** @deprecated Prefer getRequiredModels(readTtsEngine(...)) */
export const REQUIRED_MODELS: ModelSpec[] = getQwenModels();

export type ProgressEvent = Record<string, unknown>;

type HfTreeEntry = {
  type: "file" | "directory";
  path: string;
  size?: number;
};

let downloadAbort: AbortController | null = null;
let downloadActive = false;

export function isModelDownloadActive(): boolean {
  return downloadActive;
}

export function cancelModelDownload(): boolean {
  cancelRuntimeSetup();
  if (!downloadAbort) return false;
  downloadAbort.abort();
  return true;
}

export function modelFolderReady(folderPath: string): boolean {
  if (!fs.existsSync(folderPath) || !fs.statSync(folderPath).isDirectory()) return false;
  const stack = [folderPath];
  while (stack.length) {
    const dir = stack.pop()!;
    for (const name of fs.readdirSync(dir)) {
      const full = path.join(dir, name);
      let st: fs.Stats;
      try {
        st = fs.statSync(full);
      } catch {
        continue;
      }
      if (st.isDirectory()) stack.push(full);
      else if (/\.(safetensors|npz)$/i.test(name)) return true;
    }
  }
  return false;
}

export function kokoroOnnxAssetsReady(folderPath: string): boolean {
  if (!fs.existsSync(folderPath) || !fs.statSync(folderPath).isDirectory()) return false;
  const onnx = path.join(folderPath, "kokoro-v1.0.onnx");
  const voices = path.join(folderPath, "voices-v1.0.bin");
  try {
    return (
      fs.existsSync(onnx) &&
      fs.statSync(onnx).size > 1_000_000 &&
      fs.existsSync(voices) &&
      fs.statSync(voices).size > 1_000_000
    );
  } catch {
    return false;
  }
}

/** @deprecated Prefer kokoroOnnxAssetsReady / modelFolderReady by platform. */
export function kokoroAssetsReady(folderPath: string): boolean {
  if (isKokoroMlxPlatform()) {
    const mlxDir = path.join(folderPath, "Kokoro-82M-bf16");
    return modelFolderReady(mlxDir) || modelFolderReady(folderPath);
  }
  return kokoroOnnxAssetsReady(folderPath);
}

function specReady(spec: ModelSpec, folderPath: string): boolean {
  if (spec.files?.length) return kokoroOnnxAssetsReady(folderPath);
  return modelFolderReady(folderPath);
}

function kokoroRootReady(
  modelsRoot: string,
  backend: KokoroBackendId
): boolean {
  for (const spec of getKokoroModels(process.platform, backend)) {
    if (specReady(spec, modelSpecLocalDir(modelsRoot, spec))) return true;
  }
  return false;
}

/** Absolute path to Kokoro weights actually loaded by the TTS server. */
export function resolveKokoroWeightsDir(
  auraRoot: string,
  auraDataDir: string
): string {
  if (process.env.KOKORO_MODEL_DIR) {
    return path.resolve(process.env.KOKORO_MODEL_DIR);
  }
  const root = resolveModelsDir(auraRoot, auraDataDir, "kokoro");
  const spec = getKokoroModels(
    process.platform,
    readKokoroBackend(auraDataDir)
  )[0];
  if (!spec) return root;
  return modelSpecLocalDir(root, spec);
}

export function kokoroBackendId(
  platform = process.platform,
  preferred: KokoroBackendId = defaultKokoroBackend(platform)
): "mlx" | "onnx" {
  return preferred === "mlx" && isKokoroMlxPlatform(platform) ? "mlx" : "onnx";
}

export function projectModelsDir(
  auraRoot: string,
  engine: TtsEngineId = "qwen3",
  platform = process.platform
): string {
  if (engine === "kokoro") {
    return path.join(auraRoot, "tts", "kokoro", "models");
  }
  if (isTorchTtsPlatform(platform)) {
    return path.join(auraRoot, "tts", "torch", "models");
  }
  return path.join(auraRoot, "qwen3-tts-apple-silicon", "models");
}

export function resolveModelsDir(
  auraRoot: string,
  auraDataDir: string,
  engine?: TtsEngineId
): string {
  const active = engine ?? readTtsEngine(auraDataDir);
  if (active === "kokoro") {
    const backend = kokoroBackendId(
      process.platform,
      readKokoroBackend(auraDataDir)
    );
    if (process.env.KOKORO_MODEL_DIR) {
      const resolved = path.resolve(process.env.KOKORO_MODEL_DIR);
      // Allow pointing at the bf16 leaf; status/download use the parent root.
      if (
        backend === "mlx" &&
        path.basename(resolved) === "Kokoro-82M-bf16"
      ) {
        return path.dirname(resolved);
      }
      return resolved;
    }
    const projectModels = projectModelsDir(auraRoot, "kokoro");
    const dataModels = path.join(auraDataDir, "models", "kokoro");
    if (kokoroRootReady(projectModels, backend)) return projectModels;
    if (kokoroRootReady(dataModels, backend)) return dataModels;
    if (path.resolve(auraDataDir) !== path.resolve(auraRoot)) return dataModels;
    return projectModels;
  }

  if (process.env.QWEN_TTS_MODELS_DIR) {
    return path.resolve(process.env.QWEN_TTS_MODELS_DIR);
  }
  const models = getQwenModels();
  const projectModels = projectModelsDir(auraRoot, "qwen3");
  const dataModels = path.join(auraDataDir, "models");
  const projectReady = models.every((m) =>
    modelFolderReady(path.join(projectModels, m.folder))
  );
  if (projectReady) return projectModels;
  const dataReady = models.every((m) =>
    modelFolderReady(path.join(dataModels, m.folder))
  );
  if (dataReady) return dataModels;
  if (path.resolve(auraDataDir) !== path.resolve(auraRoot)) return dataModels;
  return projectModels;
}

function engineStatusBlock(
  auraRoot: string,
  auraDataDir: string,
  engine: TtsEngineId,
  kokoroBackend: KokoroBackendId
) {
  const modelsDir = resolveModelsDir(auraRoot, auraDataDir, engine);
  const models = getRequiredModels(
    engine,
    process.platform,
    kokoroBackend
  ).map((m) => {
    const folderPath = modelSpecLocalDir(modelsDir, m);
    const present = specReady(m, folderPath);
    return {
      id: m.id,
      folder: m.folder,
      label: m.label,
      present,
      approxBytes: m.approxBytes,
      path: folderPath,
    };
  });
  const runtimeReady = isEngineRuntimeReady(
    auraRoot,
    engine,
    process.platform,
    kokoroBackend
  );
  const modelsReady = models.every((m) => m.present);
  return {
    ready: modelsReady && runtimeReady,
    modelsReady,
    runtimeReady,
    modelsDir,
    models,
    voices: voicesForEngine(engine),
  };
}

export function getModelsStatus(auraRoot: string, auraDataDir: string) {
  const engine = readTtsEngine(auraDataDir);
  const kokoroDevice = readKokoroDevice(auraDataDir);
  const kokoroBackend = kokoroBackendId(
    process.platform,
    readKokoroBackend(auraDataDir)
  );
  const active = engineStatusBlock(auraRoot, auraDataDir, engine, kokoroBackend);
  const qwen3 = engineStatusBlock(auraRoot, auraDataDir, "qwen3", kokoroBackend);
  const kokoro = engineStatusBlock(auraRoot, auraDataDir, "kokoro", kokoroBackend);
  const gpu: GpuDetectResult = detectGpu(auraRoot);
  return {
    ready: active.ready,
    modelsReady: active.modelsReady,
    runtimeReady: active.runtimeReady,
    engine,
    kokoroDevice,
    kokoroBackend,
    modelsDir: active.modelsDir,
    models: active.models.map((m) => ({ ...m, downloading: downloadActive })),
    engines: {
      qwen3: {
        ready: qwen3.ready,
        modelsReady: qwen3.modelsReady,
        runtimeReady: qwen3.runtimeReady,
        modelsDir: qwen3.modelsDir,
        models: qwen3.models,
        voices: qwen3.voices,
        label: "Qwen3-TTS",
        description: "Alta qualidade com clonagem de voz (ICL).",
      },
      kokoro: {
        ready: kokoro.ready,
        modelsReady: kokoro.modelsReady,
        runtimeReady: kokoro.runtimeReady,
        modelsDir: kokoro.modelsDir,
        models: kokoro.models,
        voices: kokoro.voices,
        label: "Kokoro",
        description: kokoroBackend === "mlx"
          ? "Rápido no Apple Silicon (MLX bf16)."
          : "Qualidade máxima (ONNX fp32) com Core ML/CPU.",
      },
    },
    downloading: downloadActive,
    backend:
      engine === "kokoro"
        ? kokoroBackend
        : isTorchTtsPlatform()
          ? "torch"
          : "mlx",
    platform: process.platform,
    gpu,
    voices: active.voices,
    kokoroAccel: probeKokoroAccel(auraRoot, kokoroDevice, kokoroBackend),
  };
}

function hfHeaders(): Record<string, string> {
  const headers: Record<string, string> = {
    "User-Agent": "AuraReader/0.0.1",
  };
  const token = process.env.HF_TOKEN || process.env.HUGGING_FACE_HUB_TOKEN;
  if (token) headers.Authorization = `Bearer ${token}`;
  return headers;
}

async function listRepoFiles(repo: string, signal: AbortSignal): Promise<HfTreeEntry[]> {
  const url = `https://huggingface.co/api/models/${repo}/tree/main?recursive=true`;
  const res = await fetch(url, { headers: hfHeaders(), signal });
  if (!res.ok) {
    throw new Error(`Falha ao listar ${repo}: HTTP ${res.status}`);
  }
  const tree = (await res.json()) as HfTreeEntry[];
  return tree.filter((e) => {
    if (e.type !== "file" || !e.path) return false;
    if (e.path.endsWith(".gitattributes") || path.basename(e.path) === ".gitattributes") {
      return false;
    }
    // Kokoro MLX: skip demo samples + torch .pt duplicates (safetensors is enough).
    if (/Kokoro/i.test(repo)) {
      if (e.path.startsWith("samples/")) return false;
      if (e.path.endsWith(".pt")) return false;
    }
    return true;
  });
}

function ensureParentDir(filePath: string) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
}

async function downloadUrlToFile(
  url: string,
  destPath: string,
  expectedSize: number,
  signal: AbortSignal,
  onBytes: (received: number, total: number) => void,
  headers: Record<string, string> = { "User-Agent": "AuraReader/0.0.1" }
): Promise<number> {
  const partialPath = `${destPath}.partial`;

  let startAt = 0;
  if (fs.existsSync(partialPath)) {
    startAt = fs.statSync(partialPath).size;
  } else if (fs.existsSync(destPath) && expectedSize > 0) {
    const existing = fs.statSync(destPath).size;
    if (existing === expectedSize || (expectedSize > 1_000_000 && existing > expectedSize * 0.95)) {
      onBytes(existing, existing);
      return existing;
    }
  }

  const reqHeaders = { ...headers };
  if (startAt > 0) reqHeaders.Range = `bytes=${startAt}-`;

  const res = await fetch(url, { headers: reqHeaders, signal, redirect: "follow" });
  if (!(res.ok || res.status === 206)) {
    throw new Error(`Download falhou (${path.basename(destPath)}): HTTP ${res.status}`);
  }

  const totalHeader = res.headers.get("content-length");
  const chunkLen = totalHeader ? Number(totalHeader) : expectedSize - startAt;
  const total = expectedSize > 0 ? expectedSize : startAt + (Number.isFinite(chunkLen) ? chunkLen : 0);

  ensureParentDir(destPath);
  const flags = startAt > 0 && res.status === 206 ? "a" : "w";
  if (flags === "w" && fs.existsSync(partialPath)) fs.unlinkSync(partialPath);

  const body = res.body;
  if (!body) throw new Error(`Resposta vazia ao baixar ${path.basename(destPath)}`);

  let received = flags === "a" ? startAt : 0;
  onBytes(received, total || expectedSize);

  const nodeReadable = Readable.fromWeb(body as unknown as WebReadableStream<Uint8Array>);
  const writeStream = fs.createWriteStream(partialPath, { flags });

  nodeReadable.on("data", (chunk: Buffer | Uint8Array) => {
    received += chunk.length;
    onBytes(received, total || expectedSize || received);
  });

  await pipeline(nodeReadable, writeStream);

  if (expectedSize > 0 && received !== expectedSize) {
    if (res.status === 206 && received < expectedSize) {
      throw new Error(
        `Arquivo incompleto: ${path.basename(destPath)} (${received}/${expectedSize})`
      );
    }
  }

  fs.renameSync(partialPath, destPath);
  return received;
}

async function downloadFile(
  repo: string,
  relativePath: string,
  destPath: string,
  expectedSize: number,
  signal: AbortSignal,
  onBytes: (received: number, total: number) => void
): Promise<number> {
  const url = `https://huggingface.co/${repo}/resolve/main/${relativePath
    .split("/")
    .map(encodeURIComponent)
    .join("/")}`;
  return downloadUrlToFile(url, destPath, expectedSize, signal, onBytes, hfHeaders());
}

function formatEta(seconds: number | null): string | null {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return null;
  const s = Math.round(seconds);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const rem = s % 60;
  if (m < 60) return `${m}m ${rem}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

export async function downloadMissingModels(options: {
  auraRoot: string;
  auraDataDir: string;
  engine?: TtsEngineId;
  onEvent: (evt: ProgressEvent) => void;
}): Promise<void> {
  if (downloadActive) {
    throw new Error("Download já em andamento.");
  }

  const engine = options.engine ?? readTtsEngine(options.auraDataDir);
  const kokoroBackend = kokoroBackendId(
    process.platform,
    readKokoroBackend(options.auraDataDir)
  );
  const modelsDir = resolveModelsDir(options.auraRoot, options.auraDataDir, engine);
  fs.mkdirSync(modelsDir, { recursive: true });

  const abort = new AbortController();
  downloadAbort = abort;
  downloadActive = true;

  const startedAt = Date.now();
  let lastEmit = 0;
  let lastBytes = 0;
  let lastSpeedAt = startedAt;
  let speedEma = 0;

  const emit = (evt: ProgressEvent, force = false) => {
    const now = Date.now();
    if (!force && now - lastEmit < 200 && evt.type === "progress") return;
    lastEmit = now;
    options.onEvent(evt);
  };

  try {
    emit(
      {
        type: "start",
        modelsDir,
        engine,
        backend:
          engine === "kokoro"
            ? kokoroBackend
            : isTorchTtsPlatform()
              ? "torch"
              : "mlx",
      },
      true
    );

    // Create/repair the Python runtime (venv / onnxruntime / torch) before weights.
    await ensureEngineRuntime({
      auraRoot: options.auraRoot,
      engine,
      kokoroBackend,
      signal: abort.signal,
      onEvent: (evt) => emit(evt, true),
    });

    for (const spec of getRequiredModels(engine, process.platform, kokoroBackend)) {
      if (abort.signal.aborted) throw new Error("Download cancelado.");

      const localDir = modelSpecLocalDir(modelsDir, spec);
      if (specReady(spec, localDir)) {
        emit(
          {
            type: "model_skip",
            model: spec.id,
            label: spec.label,
            folder: spec.folder,
            reason: "already_present",
          },
          true
        );
        continue;
      }

      fs.mkdirSync(localDir, { recursive: true });

      if (spec.files?.length) {
        const totalBytes = spec.files.reduce((sum, f) => sum + f.approxBytes, 0);
        let modelDownloaded = 0;

        emit(
          {
            type: "model_start",
            model: spec.id,
            label: spec.label,
            repo: "kokoro-onnx",
            folder: spec.folder,
            totalBytes,
            files: spec.files.length,
          },
          true
        );

        for (let i = 0; i < spec.files.length; i++) {
          if (abort.signal.aborted) throw new Error("Download cancelado.");
          const file = spec.files[i];
          const size = file.approxBytes;
          const dest = path.join(localDir, file.name);

          emit(
            {
              type: "file_start",
              model: spec.id,
              label: spec.label,
              file: file.name,
              fileIndex: i + 1,
              fileCount: spec.files.length,
              fileBytes: size,
              downloadedBytes: modelDownloaded,
              totalBytes,
              percent: totalBytes ? Math.round((1000 * modelDownloaded) / totalBytes) / 10 : 0,
            },
            true
          );

          await downloadUrlToFile(
            file.url,
            dest,
            size,
            abort.signal,
            (fileReceived, fileTotal) => {
              const overallDownloaded = modelDownloaded + fileReceived;
              const now = Date.now();
              const dt = (now - lastSpeedAt) / 1000;
              if (dt >= 0.4) {
                const instant = (overallDownloaded - lastBytes) / Math.max(dt, 0.001);
                speedEma = speedEma ? speedEma * 0.7 + instant * 0.3 : instant;
                lastBytes = overallDownloaded;
                lastSpeedAt = now;
              }
              const remaining = Math.max(0, totalBytes - overallDownloaded);
              const etaSeconds = speedEma > 1024 ? remaining / speedEma : null;
              emit({
                type: "progress",
                model: spec.id,
                label: spec.label,
                file: file.name,
                fileIndex: i + 1,
                fileCount: spec.files!.length,
                fileDownloadedBytes: fileReceived,
                fileTotalBytes: fileTotal || size,
                filePercent:
                  fileTotal || size
                    ? Math.round((1000 * fileReceived) / (fileTotal || size)) / 10
                    : 0,
                downloadedBytes: overallDownloaded,
                totalBytes,
                percent: totalBytes
                  ? Math.round((1000 * overallDownloaded) / totalBytes) / 10
                  : 0,
                bytesPerSecond: Math.round(speedEma),
                etaSeconds: etaSeconds != null ? Math.round(etaSeconds) : null,
                etaLabel: formatEta(etaSeconds),
              });
            }
          );

          modelDownloaded += fs.statSync(dest).size;
          emit(
            {
              type: "file_done",
              model: spec.id,
              file: file.name,
              fileIndex: i + 1,
              fileCount: spec.files.length,
              downloadedBytes: modelDownloaded,
              totalBytes,
              percent: totalBytes
                ? Math.round((1000 * modelDownloaded) / totalBytes) / 10
                : 100,
              bytesPerSecond: Math.round(speedEma),
            },
            true
          );
        }

        emit(
          {
            type: "model_done",
            model: spec.id,
            label: spec.label,
            folder: spec.folder,
            downloadedBytes: modelDownloaded,
            totalBytes,
          },
          true
        );
        continue;
      }

      const files = await listRepoFiles(spec.repo, abort.signal);
      const totalBytes = files.reduce((sum, f) => sum + (f.size || 0), 0);
      let modelDownloaded = 0;

      emit(
        {
          type: "model_start",
          model: spec.id,
          label: spec.label,
          repo: spec.repo,
          folder: spec.folder,
          totalBytes,
          files: files.length,
        },
        true
      );

      for (let i = 0; i < files.length; i++) {
        if (abort.signal.aborted) throw new Error("Download cancelado.");
        const file = files[i];
        const size = file.size || 0;
        const dest = path.join(localDir, file.path);

        emit(
          {
            type: "file_start",
            model: spec.id,
            label: spec.label,
            file: file.path,
            fileIndex: i + 1,
            fileCount: files.length,
            fileBytes: size,
            downloadedBytes: modelDownloaded,
            totalBytes,
            percent: totalBytes ? Math.round((1000 * modelDownloaded) / totalBytes) / 10 : 0,
          },
          true
        );

        await downloadFile(
          spec.repo,
          file.path,
          dest,
          size,
          abort.signal,
          (fileReceived, fileTotal) => {
            const overallDownloaded = modelDownloaded + fileReceived;
            const now = Date.now();
            const dt = (now - lastSpeedAt) / 1000;
            if (dt >= 0.4) {
              const instant = (overallDownloaded - lastBytes) / Math.max(dt, 0.001);
              speedEma = speedEma ? speedEma * 0.7 + instant * 0.3 : instant;
              lastBytes = overallDownloaded;
              lastSpeedAt = now;
            }
            const remaining = Math.max(0, totalBytes - overallDownloaded);
            const etaSeconds = speedEma > 1024 ? remaining / speedEma : null;

            emit({
              type: "progress",
              model: spec.id,
              label: spec.label,
              file: file.path,
              fileIndex: i + 1,
              fileCount: files.length,
              fileDownloadedBytes: fileReceived,
              fileTotalBytes: fileTotal || size,
              filePercent:
                fileTotal || size
                  ? Math.round((1000 * fileReceived) / (fileTotal || size)) / 10
                  : 0,
              downloadedBytes: overallDownloaded,
              totalBytes,
              percent: totalBytes
                ? Math.round((1000 * overallDownloaded) / totalBytes) / 10
                : 0,
              bytesPerSecond: Math.round(speedEma),
              etaSeconds: etaSeconds != null ? Math.round(etaSeconds) : null,
              etaLabel: formatEta(etaSeconds),
            });
          }
        );

        modelDownloaded += size || fs.statSync(dest).size;
        emit(
          {
            type: "file_done",
            model: spec.id,
            file: file.path,
            fileIndex: i + 1,
            fileCount: files.length,
            downloadedBytes: modelDownloaded,
            totalBytes,
            percent: totalBytes ? Math.round((1000 * modelDownloaded) / totalBytes) / 10 : 100,
            bytesPerSecond: Math.round(speedEma),
            etaSeconds:
              speedEma > 1024 ? Math.round((totalBytes - modelDownloaded) / speedEma) : null,
            etaLabel: formatEta(
              speedEma > 1024 ? (totalBytes - modelDownloaded) / speedEma : null
            ),
          },
          true
        );
      }

      emit(
        {
          type: "model_done",
          model: spec.id,
          label: spec.label,
          folder: spec.folder,
          downloadedBytes: modelDownloaded,
          totalBytes,
        },
        true
      );
    }

    const status = getModelsStatus(options.auraRoot, options.auraDataDir);
    emit({ type: "done", ready: status.ready, modelsDir: status.modelsDir, engine }, true);
    if (!status.ready && status.engine === engine) {
      if (!status.runtimeReady) {
        throw new Error("Download terminou, mas o runtime TTS ainda está incompleto.");
      }
      throw new Error("Download terminou, mas os modelos ainda estão incompletos.");
    }
  } catch (err: any) {
    const cancelled =
      abort.signal.aborted ||
      err?.name === "AbortError" ||
      /cancelad/i.test(String(err?.message || err));
    emit(
      {
        type: cancelled ? "cancelled" : "error",
        message: cancelled ? "Download cancelado." : err?.message || String(err),
        ready: getModelsStatus(options.auraRoot, options.auraDataDir).ready,
      },
      true
    );
    if (!cancelled) throw err;
  } finally {
    downloadActive = false;
    downloadAbort = null;
  }
}

export function deleteModels(
  auraRoot: string,
  auraDataDir: string,
  ids?: string[],
  engine?: TtsEngineId
): { deleted: string[]; modelsDir: string } {
  if (downloadActive) {
    throw new Error("Cancele o download antes de excluir modelos.");
  }
  const active = engine ?? readTtsEngine(auraDataDir);
  const kokoroBackend = kokoroBackendId(
    process.platform,
    readKokoroBackend(auraDataDir)
  );
  const modelsDir = resolveModelsDir(auraRoot, auraDataDir, active);
  const all = getRequiredModels(active, process.platform, kokoroBackend);
  const targets = ids && ids.length ? all.filter((m) => ids.includes(m.id)) : all;

  const deleted: string[] = [];
  for (const spec of targets) {
    const folderPath = modelSpecLocalDir(modelsDir, spec);
    if (fs.existsSync(folderPath)) {
      if (spec.files?.length) {
        for (const f of spec.files) {
          const p = path.join(folderPath, f.name);
          if (fs.existsSync(p)) fs.rmSync(p, { force: true });
          if (fs.existsSync(`${p}.partial`)) fs.rmSync(`${p}.partial`, { force: true });
        }
        deleted.push(spec.id);
      } else {
        fs.rmSync(folderPath, { recursive: true, force: true });
        deleted.push(spec.id);
      }
    }
    const partialRoot = `${folderPath}.partial`;
    if (fs.existsSync(partialRoot)) {
      fs.rmSync(partialRoot, { recursive: true, force: true });
    }
  }
  return { deleted, modelsDir };
}
