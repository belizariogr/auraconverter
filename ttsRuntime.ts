/**
 * Detect / prepare TTS Python runtimes (Kokoro ONNX/MLX, Qwen Torch/MLX).
 * Called during model install so users don't need a separate CLI setup step.
 */
import { spawn, type ChildProcess } from "child_process";
import fs from "fs";
import path from "path";
import {
  defaultKokoroBackend,
  type KokoroBackendId,
  type TtsEngineId,
} from "./ttsEngine";

export type RuntimeProgressEvent = Record<string, unknown>;

let activeSetupChild: ChildProcess | null = null;

export function cancelRuntimeSetup(): boolean {
  if (!activeSetupChild || activeSetupChild.killed) return false;
  try {
    activeSetupChild.kill("SIGTERM");
  } catch {
    // ignore
  }
  return true;
}

function hasPythonAt(...candidates: string[]): boolean {
  return candidates.some((p) => fs.existsSync(p));
}

function hasMisakiMarker(auraRoot: string): boolean {
  const candidates = [
    path.join(auraRoot, "mlx", "site-packages", "misaki"),
    path.join(
      auraRoot,
      "mlx",
      ".venv",
      "lib",
      "python3.12",
      "site-packages",
      "misaki"
    ),
  ];
  return candidates.some((p) => fs.existsSync(p));
}

export function isKokoroRuntimeReady(
  auraRoot: string,
  platform = process.platform,
  backend: KokoroBackendId = defaultKokoroBackend(platform)
): boolean {
  const ttsDir = path.join(auraRoot, "tts", "kokoro");
  if (backend === "mlx" && platform === "darwin") {
    if (!fs.existsSync(path.join(ttsDir, "tts_server_mlx.py"))) return false;
    // Shared MLX stack with Qwen; misaki is required for Kokoro G2P.
    return isQwenMlxRuntimeReady(auraRoot) && hasMisakiMarker(auraRoot);
  }
  if (!fs.existsSync(path.join(ttsDir, "tts_server.py"))) return false;
  if (fs.existsSync(path.join(ttsDir, "site-packages"))) return true;
  return hasPythonAt(
    path.join(ttsDir, ".venv", "bin", "python"),
    path.join(ttsDir, ".venv", "Scripts", "python.exe")
  );
}

export function isQwenTorchRuntimeReady(auraRoot: string): boolean {
  const ttsDir = path.join(auraRoot, "tts", "torch");
  if (!fs.existsSync(path.join(ttsDir, "tts_server.py"))) return false;
  if (fs.existsSync(path.join(ttsDir, "site-packages"))) return true;
  return hasPythonAt(
    path.join(ttsDir, ".venv", "bin", "python"),
    path.join(ttsDir, ".venv", "Scripts", "python.exe")
  );
}

export function isQwenMlxRuntimeReady(auraRoot: string): boolean {
  const ttsDir = path.join(auraRoot, "mlx");
  if (!fs.existsSync(path.join(ttsDir, "tts_server.py"))) return false;
  if (fs.existsSync(path.join(ttsDir, "site-packages"))) return true;
  return hasPythonAt(path.join(ttsDir, ".venv", "bin", "python"));
}

export function isEngineRuntimeReady(
  auraRoot: string,
  engine: TtsEngineId,
  platform = process.platform,
  kokoroBackend: KokoroBackendId = defaultKokoroBackend(platform)
): boolean {
  if (engine === "kokoro") {
    return isKokoroRuntimeReady(auraRoot, platform, kokoroBackend);
  }
  if (platform !== "darwin") {
    return false;
  }
  return isQwenMlxRuntimeReady(auraRoot);
}

function setupScriptPath(auraRoot: string, name: string): string | null {
  const candidates = [
    path.join(auraRoot, "scripts", name),
    // Dev when AURA_ROOT points at a nested data dir / packaged layout misses scripts.
    path.join(process.cwd(), "scripts", name),
  ];
  return candidates.find((p) => fs.existsSync(p)) || null;
}

function nodeBinary(): string {
  // Prefer the current interpreter when it's Node/Bun (not Electron's helper binary).
  const exe = process.execPath || "";
  if (exe && !/electron/i.test(exe)) return exe;
  return "node";
}

async function runSetupScript(options: {
  scriptPath: string;
  label: string;
  args?: string[];
  onEvent: (evt: RuntimeProgressEvent) => void;
  signal?: AbortSignal;
}): Promise<void> {
  const { scriptPath, label, args = [], onEvent, signal } = options;

  if (signal?.aborted) throw new Error("Preparação cancelada.");

  onEvent({
    type: "runtime_start",
    label,
    phase: `Preparando runtime ${label}…`,
  });

  await new Promise<void>((resolve, reject) => {
    const child = spawn(nodeBinary(), [scriptPath, ...args], {
      cwd: path.dirname(path.dirname(scriptPath)),
      env: { ...process.env },
      stdio: ["ignore", "pipe", "pipe"],
    });
    activeSetupChild = child;

    let stderrTail = "";
    const onAbort = () => {
      try {
        child.kill("SIGTERM");
      } catch {
        // ignore
      }
    };
    signal?.addEventListener("abort", onAbort, { once: true });

    const emitLine = (line: string, stream: "stdout" | "stderr") => {
      const text = line.trim();
      if (!text) return;
      if (stream === "stderr") {
        stderrTail = (stderrTail + "\n" + text).slice(-2000);
      }
      onEvent({
        type: "runtime_log",
        label,
        stream,
        line: text.slice(0, 400),
        phase: text.slice(0, 120),
      });
    };

    const attach = (stream: NodeJS.ReadableStream | null, name: "stdout" | "stderr") => {
      if (!stream) return;
      let buf = "";
      stream.setEncoding("utf8");
      stream.on("data", (chunk: string) => {
        buf += chunk;
        const parts = buf.split(/\r?\n/);
        buf = parts.pop() || "";
        for (const line of parts) emitLine(line, name);
      });
      stream.on("end", () => {
        if (buf.trim()) emitLine(buf, name);
      });
    };

    attach(child.stdout, "stdout");
    attach(child.stderr, "stderr");

    child.on("error", (err) => {
      signal?.removeEventListener("abort", onAbort);
      if (activeSetupChild === child) activeSetupChild = null;
      reject(err);
    });

    child.on("close", (code, signalName) => {
      signal?.removeEventListener("abort", onAbort);
      if (activeSetupChild === child) activeSetupChild = null;
      if (signal?.aborted || signalName === "SIGTERM" || signalName === "SIGINT") {
        reject(new Error("Preparação cancelada."));
        return;
      }
      if (code !== 0) {
        reject(
          new Error(
            `Falha ao preparar ${label} (exit ${code}).` +
              (stderrTail ? `\n${stderrTail.trim()}` : "")
          )
        );
        return;
      }
      resolve();
    });
  });

  onEvent({
    type: "runtime_done",
    label,
    phase: `Runtime ${label} pronto`,
  });
}

async function ensureMlxRuntime(options: {
  auraRoot: string;
  onEvent: (evt: RuntimeProgressEvent) => void;
  signal?: AbortSignal;
}): Promise<void> {
  const { auraRoot, onEvent, signal } = options;
  const script = setupScriptPath(auraRoot, "setup-mlx-tts.cjs");
  if (!script) {
    throw new Error(
      "Runtime MLX ausente e scripts/setup-mlx-tts.cjs não encontrado. " +
        "Em desenvolvimento: bun run setup:tts:mlx"
    );
  }
  await runSetupScript({
    scriptPath: script,
    label: "Breeze TTS 2 (MLX)",
    onEvent,
    signal,
  });
  if (!isQwenMlxRuntimeReady(auraRoot)) {
    throw new Error("Runtime MLX ainda incompleto após a instalação.");
  }
}

/**
 * Ensure the Python runtime for the active engine exists.
 * Packaged builds already ship site-packages → no-op.
 * Dev checkouts run setup-kokoro-tts / setup-torch-tts (or MLX venv) as needed.
 */
export async function ensureEngineRuntime(options: {
  auraRoot: string;
  engine: TtsEngineId;
  kokoroBackend?: KokoroBackendId;
  onEvent: (evt: RuntimeProgressEvent) => void;
  signal?: AbortSignal;
}): Promise<void> {
  const { auraRoot, engine, onEvent, signal } = options;
  const kokoroBackend =
    options.kokoroBackend ?? defaultKokoroBackend(process.platform);

  if (isEngineRuntimeReady(auraRoot, engine, process.platform, kokoroBackend)) {
    onEvent({
      type: "runtime_skip",
      engine,
      reason: "already_present",
      phase: "Runtime já preparado",
    });
    return;
  }

  if (engine === "kokoro") {
    if (kokoroBackend === "mlx" && process.platform === "darwin") {
      await ensureMlxRuntime({ auraRoot, onEvent, signal });
      if (!isKokoroRuntimeReady(auraRoot, process.platform, kokoroBackend)) {
        throw new Error(
          "Runtime Kokoro (MLX) ainda incompleto após a preparação. " +
            "Confirme misaki[en] em mlx/requirements.txt."
        );
      }
      return;
    }

    const script = setupScriptPath(auraRoot, "setup-kokoro-tts.cjs");
    if (!script) {
      throw new Error(
        "Runtime Kokoro ausente e scripts/setup-kokoro-tts.cjs não encontrado. " +
          "Em desenvolvimento: bun run setup:tts:kokoro"
      );
    }
    await runSetupScript({
      scriptPath: script,
      label: "Kokoro",
      args: ["--accel=auto"],
      onEvent,
      signal,
    });
    if (!isKokoroRuntimeReady(auraRoot, process.platform, kokoroBackend)) {
      throw new Error("Runtime Kokoro ainda incompleto após a preparação.");
    }
    return;
  }

  if (process.platform !== "darwin") {
    throw new Error("O Breeze TTS 2 requer macOS Apple Silicon com MLX.");
  }

  await ensureMlxRuntime({ auraRoot, onEvent, signal });
}
