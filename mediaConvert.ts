import fs from "fs";
import path from "path";
import os from "os";
import { spawn } from "child_process";
import { ffmpegBin, ffprobeBin } from "./ffmpegBin";

let ffmpegAvailable: boolean | null = null;

export async function ensureFfmpeg(): Promise<void> {
  if (ffmpegAvailable === true) return;
  const bin = ffmpegBin();
  await new Promise<void>((resolve, reject) => {
    const child = spawn(bin, ["-version"], { stdio: ["ignore", "pipe", "pipe"] });
    child.on("error", () => {
      ffmpegAvailable = false;
      reject(
        new Error(
          "ffmpeg não encontrado. No app Electron ele vem embutido; em desenvolvimento instale com `brew install ffmpeg` (macOS) ou o equivalente do seu sistema."
        )
      );
    });
    child.on("close", (code) => {
      if (code === 0) {
        ffmpegAvailable = true;
        resolve();
      } else {
        ffmpegAvailable = false;
        reject(new Error("ffmpeg não está disponível neste sistema."));
      }
    });
  });
}

function tmpMediaDir(): string {
  const dir = path.join(os.tmpdir(), "aura-media");
  fs.mkdirSync(dir, { recursive: true });
  return dir;
}

export function uniqueMediaPath(ext: string): string {
  return path.join(
    tmpMediaDir(),
    `media-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}.${ext}`
  );
}

export type FfmpegProgress = {
  /** 0–100 when duration is known; otherwise null */
  percent: number | null;
  timeSec: number;
  totalSec: number | null;
};

/** Probe media duration in seconds via ffprobe. */
export async function probeDurationSeconds(filePath: string): Promise<number | null> {
  return new Promise((resolve) => {
    const child = spawn(
      ffprobeBin(),
      ["-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", filePath],
      { stdio: ["ignore", "pipe", "pipe"] }
    );
    let out = "";
    child.stdout?.on("data", (d) => {
      out += String(d);
    });
    child.on("error", () => resolve(null));
    child.on("close", () => {
      const n = parseFloat(out.trim());
      resolve(Number.isFinite(n) && n > 0 ? n : null);
    });
  });
}

function parseProgressBlock(
  block: string,
  totalSec: number | null
): FfmpegProgress | null {
  const lines = block.split(/\r?\n/);
  let outTimeUs: number | null = null;
  let outTimeMs: number | null = null;
  let outTime: string | null = null;

  for (const line of lines) {
    const [key, ...rest] = line.split("=");
    const val = rest.join("=").trim();
    if (key === "out_time_us") outTimeUs = Number(val);
    else if (key === "out_time_ms") outTimeMs = Number(val);
    else if (key === "out_time") outTime = val;
  }

  let timeSec = 0;
  if (outTimeUs != null && Number.isFinite(outTimeUs)) {
    timeSec = outTimeUs / 1_000_000;
  } else if (outTimeMs != null && Number.isFinite(outTimeMs)) {
    // Some builds report microseconds in out_time_ms
    timeSec = outTimeMs > 1_000_000 ? outTimeMs / 1_000_000 : outTimeMs / 1000;
  } else if (outTime) {
    const m = outTime.match(/(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)/);
    if (m) {
      const h = Number(m[1] || 0);
      const min = Number(m[2] || 0);
      const s = Number(m[3] || 0);
      timeSec = h * 3600 + min * 60 + s;
    }
  } else {
    return null;
  }

  let percent: number | null = null;
  if (totalSec != null && totalSec > 0) {
    percent = Math.min(99.5, Math.max(0, (timeSec / totalSec) * 100));
  }

  return { percent, timeSec, totalSec };
}

function runFfmpeg(
  args: string[],
  opts?: {
    totalSec?: number | null;
    onProgress?: (p: FfmpegProgress) => void;
    signal?: AbortSignal;
  }
): Promise<void> {
  return new Promise((resolve, reject) => {
    if (opts?.signal?.aborted) {
      reject(new DOMException("Conversão cancelada.", "AbortError"));
      return;
    }

    const child = spawn(
      ffmpegBin(),
      ["-y", "-nostats", "-progress", "pipe:1", ...args],
      { stdio: ["ignore", "pipe", "pipe"] }
    );
    let stderr = "";
    let stdoutBuf = "";
    let lastEmit = 0;
    let settled = false;

    const cleanup = () => {
      opts?.signal?.removeEventListener("abort", onAbort);
    };

    const onAbort = () => {
      try {
        child.kill("SIGKILL");
      } catch {
        // ignore
      }
    };
    opts?.signal?.addEventListener("abort", onAbort);

    const emitProgress = (block: string) => {
      if (!opts?.onProgress) return;
      const parsed = parseProgressBlock(block, opts.totalSec ?? null);
      if (!parsed) return;
      const now = Date.now();
      if (now - lastEmit < 200 && parsed.percent != null && parsed.percent < 99) return;
      lastEmit = now;
      opts.onProgress(parsed);
    };

    child.stdout?.on("data", (d) => {
      stdoutBuf += String(d);
      const parts = stdoutBuf.split(/\n(?=progress=)/);
      stdoutBuf = parts.pop() || "";
      for (const part of parts) {
        emitProgress(part);
      }
      if (stdoutBuf.includes("\n\n")) {
        const chunks = stdoutBuf.split(/\n\n+/);
        stdoutBuf = chunks.pop() || "";
        for (const c of chunks) emitProgress(c);
      }
    });

    child.stderr?.on("data", (d) => {
      stderr += String(d);
    });
    child.on("error", (err) => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(err);
    });
    child.on("close", (code) => {
      if (settled) return;
      settled = true;
      cleanup();
      if (stdoutBuf.trim()) emitProgress(stdoutBuf);
      if (opts?.signal?.aborted) {
        reject(new DOMException("Conversão cancelada.", "AbortError"));
        return;
      }
      if (opts?.onProgress && opts.totalSec != null) {
        opts.onProgress({ percent: 100, timeSec: opts.totalSec, totalSec: opts.totalSec });
      }
      if (code === 0) resolve();
      else reject(new Error(`ffmpeg falhou (código ${code}): ${stderr.slice(-1200)}`));
    });
  });
}

function pcmDurationSeconds(pcmPath: string, sampleRate: number): Promise<number> {
  return fs.promises.stat(pcmPath).then((stat) => stat.size / (sampleRate * 2));
}

/** Books/QuickTime treat extra JPEG tracks as video. Mux a single front cover. */
function frontCoverPaths(paths: string[] | undefined): string[] {
  const found = (paths || []).find((p) => p && fs.existsSync(p));
  return found ? [found] : [];
}

function pushCoverEncodeArgs(args: string[]): void {
  args.push(
    "-c:v",
    "mjpeg",
    "-disposition:v:0",
    "attached_pic",
    "-metadata:s:v:0",
    "title=Cover",
    "-metadata:s:v:0",
    "comment=Cover (front)"
  );
}

/**
 * MP3/AAC use fixed block sizes in samples, so at 24 kHz each block spans twice
 * as much time as at 48 kHz and plosives ("t", "k") smear into a "ch"-like
 * pre-echo. Upsampling 2x before the codec halves the block duration and keeps
 * transients crisp; it adds no fake frequency content.
 */
const ENCODE_SAMPLE_RATE = 48000;

export async function pcmToMp3(opts: {
  pcmPath: string;
  outputPath?: string;
  sampleRate?: number;
  onProgress?: (p: FfmpegProgress) => void;
  signal?: AbortSignal;
}): Promise<string> {
  await ensureFfmpeg();
  const out = opts.outputPath || uniqueMediaPath("mp3");
  const sampleRate = opts.sampleRate || 24000;
  const totalSec = await pcmDurationSeconds(opts.pcmPath, sampleRate);

  await runFfmpeg(
    [
      "-f",
      "s16le",
      "-ar",
      String(sampleRate),
      "-ac",
      "1",
      "-i",
      opts.pcmPath,
      "-af",
      `aresample=${ENCODE_SAMPLE_RATE}:filter_size=64:cutoff=0.97`,
      "-ac",
      "1",
      "-c:a",
      "libmp3lame",
      "-q:a",
      "0",
      "-compression_level",
      "0",
      "-write_xing",
      "1",
      out,
    ],
    { totalSec, onProgress: opts.onProgress, signal: opts.signal }
  );
  return out;
}

export async function pcmToM4b(opts: {
  pcmPath: string;
  outputPath?: string;
  sampleRate?: number;
  artworkPaths?: string[];
  title?: string;
  onProgress?: (p: FfmpegProgress) => void;
  signal?: AbortSignal;
}): Promise<string> {
  await ensureFfmpeg();
  const out = opts.outputPath || uniqueMediaPath("m4b");
  const sampleRate = opts.sampleRate || 24000;
  const totalSec = await pcmDurationSeconds(opts.pcmPath, sampleRate);
  const artworks = frontCoverPaths(opts.artworkPaths);
  const args = [
    "-f",
    "s16le",
    "-ar",
    String(sampleRate),
    "-ac",
    "1",
    "-i",
    opts.pcmPath,
  ];
  for (const artwork of artworks) args.push("-i", artwork);

  args.push("-map", "0:a");
  for (let i = 0; i < artworks.length; i++) args.push("-map", `${i + 1}:v`);
  args.push(
    "-af",
    `aresample=${ENCODE_SAMPLE_RATE}:filter_size=64:cutoff=0.97`,
    "-ac",
    "1",
    "-c:a",
    "aac",
    "-profile:a",
    "aac_low",
    "-b:a",
    "160k",
    "-aac_coder",
    "twoloop"
  );
  if (artworks.length > 0) pushCoverEncodeArgs(args);
  if (opts.title) args.push("-metadata", `title=${opts.title}`);
  args.push("-movflags", "+faststart", "-f", "mp4", out);

  await runFfmpeg(args, { totalSec, onProgress: opts.onProgress, signal: opts.signal });
  return out;
}

/**
 * Convert MP3 to M4B (AAC in MP4) and attach artwork images as attached_pic streams.
 * First image is treated as the primary cover.
 */
export async function mp3ToM4b(opts: {
  mp3Path: string;
  outputPath?: string;
  artworkPaths?: string[];
  title?: string;
  onProgress?: (p: FfmpegProgress) => void;
  signal?: AbortSignal;
}): Promise<string> {
  await ensureFfmpeg();
  const out = opts.outputPath || uniqueMediaPath("m4b");
  const artworks = frontCoverPaths(opts.artworkPaths);
  const totalSec = await probeDurationSeconds(opts.mp3Path);

  const args: string[] = ["-i", opts.mp3Path];
  for (const art of artworks) {
    args.push("-i", art);
  }

  // Map audio from first input
  args.push("-map", "0:a");
  // Map each artwork as video stream
  for (let i = 0; i < artworks.length; i++) {
    args.push("-map", `${i + 1}:v`);
  }

  args.push("-c:a", "aac", "-b:a", "192k", "-aac_coder", "twoloop");
  if (artworks.length > 0) {
    pushCoverEncodeArgs(args);
  }

  if (opts.title) {
    args.push("-metadata", `title=${opts.title}`);
  }

  args.push("-movflags", "+faststart", "-f", "mp4", out);
  await runFfmpeg(args, {
    totalSec,
    onProgress: opts.onProgress,
    signal: opts.signal,
  });
  return out;
}

/**
 * Demux M4B/M4A: extract audio to MP3 and first attached cover to JPEG (if any).
 */
export async function m4bToMp3AndCover(opts: {
  m4bPath: string;
  mp3Path?: string;
  coverPath?: string;
  onProgress?: (p: FfmpegProgress) => void;
  signal?: AbortSignal;
}): Promise<{ mp3Path: string; coverPath: string | null }> {
  await ensureFfmpeg();
  const mp3Path = opts.mp3Path || uniqueMediaPath("mp3");
  const coverPath = opts.coverPath || uniqueMediaPath("jpg");
  const totalSec = await probeDurationSeconds(opts.m4bPath);

  await runFfmpeg(
    [
      "-i",
      opts.m4bPath,
      "-vn",
      "-c:a",
      "libmp3lame",
      "-q:a",
      "0",
      "-compression_level",
      "0",
      "-write_xing",
      "1",
      mp3Path,
    ],
    { totalSec, onProgress: opts.onProgress, signal: opts.signal }
  );

  if (opts.signal?.aborted) {
    throw new DOMException("Conversão cancelada.", "AbortError");
  }

  let coverOk: string | null = null;
  try {
    await runFfmpeg([
      "-i",
      opts.m4bPath,
      "-an",
      "-map",
      "0:v:0",
      "-frames:v",
      "1",
      "-q:v",
      "2",
      coverPath,
    ]);
    if (fs.existsSync(coverPath) && (await fs.promises.stat(coverPath)).size > 0) {
      coverOk = coverPath;
    }
  } catch {
    await fs.promises.unlink(coverPath).catch(() => undefined);
  }

  return { mp3Path, coverPath: coverOk };
}

export async function writeBufferToTemp(buf: Buffer, ext: string): Promise<string> {
  const p = uniqueMediaPath(ext);
  await fs.promises.writeFile(p, buf);
  return p;
}
