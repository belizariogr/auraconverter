import fs from "fs";
import path from "path";

type ConsoleMethod = "log" | "info" | "warn" | "error" | "debug";

let logFilePath: string | null = null;
let writeStream: fs.WriteStream | null = null;
let installed = false;

export function getBackendLogPath(): string | null {
  return logFilePath;
}

/**
 * Tee console.* into AURA_DATA_DIR/logs/backend-YYYY-MM-DD.log.
 * Safe to call once at server boot; subsequent calls are no-ops for console wrapping.
 */
export function installFileLogging(dataDir: string): string {
  const logsDir = path.join(dataDir, "logs");
  fs.mkdirSync(logsDir, { recursive: true });

  const day = new Date().toISOString().slice(0, 10);
  logFilePath = path.join(logsDir, `backend-${day}.log`);

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
