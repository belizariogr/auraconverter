import fs from "fs";
import path from "path";

type ConsoleMethod = "log" | "info" | "warn" | "error" | "debug";

let logsDirPath: string | null = null;
let logFilePath: string | null = null;
let writeStream: fs.WriteStream | null = null;
let installed = false;

export function getBackendLogPath(): string | null {
  return logFilePath;
}

export function getLogsDir(): string | null {
  return logsDirPath;
}

export type FailedNarrationChunkDump = {
  docId: string | null;
  /** 0-based index in the chunk list. */
  chunkIndex: number;
  /** 1-based part number shown in the UI. */
  partNum: number;
  totalChunks: number;
  voice: string;
  engine: string;
  language: string;
  reason: string;
  attempts: number;
  text: string;
};

/**
 * Persist the failed block text + metadata under logs/failed-chunks/ for debugging.
 * Returns the JSON path, or null if logging is not installed.
 */
export function saveFailedNarrationChunk(
  dump: FailedNarrationChunkDump
): string | null {
  if (!logsDirPath) {
    return null;
  }

  const failedDir = path.join(logsDirPath, "failed-chunks");
  fs.mkdirSync(failedDir, { recursive: true });

  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const docTag = dump.docId ? dump.docId.slice(0, 8) : "nodoc";
  const base = `${stamp}-chunk-${String(dump.partNum).padStart(4, "0")}-${docTag}`;
  const jsonPath = path.join(failedDir, `${base}.json`);
  const txtPath = path.join(failedDir, `${base}.txt`);

  const payload = {
    savedAt: new Date().toISOString(),
    docId: dump.docId,
    chunkIndex: dump.chunkIndex,
    partNum: dump.partNum,
    totalChunks: dump.totalChunks,
    voice: dump.voice,
    engine: dump.engine,
    language: dump.language,
    reason: dump.reason,
    attempts: dump.attempts,
    textLength: dump.text.length,
    text: dump.text,
  };

  fs.writeFileSync(jsonPath, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
  fs.writeFileSync(txtPath, dump.text, "utf8");

  return jsonPath;
}

/**
 * Tee console.* into AURA_DATA_DIR/logs/backend-YYYY-MM-DD.log.
 * Safe to call once at server boot; subsequent calls are no-ops for console wrapping.
 */
export function installFileLogging(dataDir: string): string {
  logsDirPath = path.join(dataDir, "logs");
  fs.mkdirSync(logsDirPath, { recursive: true });

  const day = new Date().toISOString().slice(0, 10);
  logFilePath = path.join(logsDirPath, `backend-${day}.log`);

  if (!writeStream) {
    writeStream = fs.createWriteStream(logFilePath, { flags: "a" });
    writeStream.on("error", (err) => {
      process.stderr.write(`[fileLog] write failed: ${err.message}\n`);
    });
  }

  if (!installed) {
    installed = true;
    const methods: ConsoleMethod[] = ["log", "info", "warn", "error", "debug"];

    for (const method of methods) {
      const original = console[method].bind(console);
      console[method] = (...args: unknown[]) => {
        original(...args);
        appendLine(method, args);
      };
    }
  }

  appendLine("info", [`[fileLog] Gravando logs em ${logFilePath}`]);
  return logFilePath;
}

function formatArg(arg: unknown): string {
  if (typeof arg === "string") {
    return arg;
  }

  if (arg instanceof Error) {
    return arg.stack || arg.message;
  }

  try {
    return JSON.stringify(arg);
  } catch {
    return String(arg);
  }
}

function appendLine(level: string, args: unknown[]): void {
  if (!writeStream) {
    return;
  }

  const stamp = new Date().toISOString();
  const body = args.map(formatArg).join(" ");
  writeStream.write(`${stamp} [${level}] ${body}\n`);
}
