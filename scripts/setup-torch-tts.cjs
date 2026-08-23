/**
 * Create tts/torch/.venv with portable CPython 3.12 + qwen-tts + torch.
 *
 *   bun run setup:tts              # auto: cuda|rocm|cpu from GPU detect
 *   bun run setup:tts -- --accel=rocm
 *   bun run setup:tts -- --accel=cpu
 */
const fs = require("fs");
const path = require("path");
const https = require("https");
const http = require("http");
const { execFileSync, spawnSync } = require("child_process");

const root = path.resolve(__dirname, "..");
const torchSrc = path.join(root, "tts", "torch");
const cacheDir = path.join(root, "build", "cache");

const PBS_TAG = "20260303";
const PBS_VERSION = "3.12.13";

const ROCM_WINDOWS_WHEELS = [
  "https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torch-2.9.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl",
  "https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torchaudio-2.9.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl",
  "https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torchvision-0.24.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl",
];

const TORCH_INDEX = {
  cuda: "https://download.pytorch.org/whl/cu124",
  cpu: "https://download.pytorch.org/whl/cpu",
  rocm: "https://download.pytorch.org/whl/rocm6.3",
};

function projectCacheEnv() {
  const pipCache = path.join(cacheDir, "pip");
  const tmp = path.join(cacheDir, "tmp");
  const hf = path.join(cacheDir, "huggingface");
  fs.mkdirSync(pipCache, { recursive: true });
  fs.mkdirSync(tmp, { recursive: true });
  fs.mkdirSync(hf, { recursive: true });
  return {
    ...process.env,
    PIP_DISABLE_PIP_VERSION_CHECK: "1",
    PIP_CACHE_DIR: pipCache,
    TMPDIR: tmp,
    TEMP: tmp,
    TMP: tmp,
    XDG_CACHE_HOME: cacheDir,
    HF_HOME: hf,
    HUGGINGFACE_HUB_CACHE: hf,
    GIT_TERMINAL_PROMPT: "0",
  };
}

function parseArgs(argv) {
  let accel = null;
  let force = false;
  for (const arg of argv) {
    if (arg.startsWith("--accel=")) accel = arg.slice("--accel=".length);
    if (arg === "--force") force = true;
  }
  if (accel && !["cuda", "rocm", "cpu"].includes(accel)) {
    throw new Error(`Invalid --accel=${accel} (use cuda|rocm|cpu)`);
  }
  return { accel, force };
}

function detectAccel() {
  try {
    // Prefer compiled JS via bun/node importing TS is awkward from cjs — shell out to bun.
    const out = execFileSync(
      "bun",
      [
        "-e",
        'import { detectGpu } from "./gpuDetect.ts"; const g = detectGpu(); console.log(g.recommendedAccel || "cpu");',
      ],
      { cwd: root, encoding: "utf8" }
    ).trim();
    if (out === "cuda" || out === "rocm" || out === "cpu") return out;
  } catch {
    // ignore
  }
  return "cpu";
}

function downloadFile(url, dest) {
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  if (fs.existsSync(dest) && fs.statSync(dest).size > 1_000_000) {
    console.log("[setup-torch-tts] Using cached", dest);
    return Promise.resolve();
  }
  console.log("[setup-torch-tts] Downloading", url);
  const tmp = `${dest}.partial`;
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(tmp);
    const get = (u, redirects = 0) => {
      const lib = u.startsWith("https") ? https : http;
      lib
        .get(u, (res) => {
          if (
            res.statusCode >= 300 &&
            res.statusCode < 400 &&
            res.headers.location &&
            redirects < 8
          ) {
            res.resume();
            get(res.headers.location, redirects + 1);
            return;
          }
          if (res.statusCode !== 200) {
            reject(new Error(`Download failed HTTP ${res.statusCode}: ${u}`));
            return;
          }
          res.pipe(file);
          file.on("finish", () => {
            file.close(() => {
              fs.renameSync(tmp, dest);
              resolve();
            });
          });
        })
        .on("error", (err) => {
          try {
            fs.unlinkSync(tmp);
          } catch {
            // ignore
          }
          reject(err);
        });
    };
    get(url);
  });
}

function pbsAssetName() {
  const arch = process.arch === "arm64" ? "aarch64" : "x86_64";
  if (process.platform === "win32") {
    return `cpython-${PBS_VERSION}+${PBS_TAG}-${arch}-pc-windows-msvc-install_only.tar.gz`;
  }
  return `cpython-${PBS_VERSION}+${PBS_TAG}-${arch}-unknown-linux-gnu-install_only.tar.gz`;
}

async function ensurePortablePython() {
  const asset = pbsAssetName();
  const url = `https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_TAG}/${asset}`;
  const archive = path.join(cacheDir, asset);
  const extractRoot = path.join(cacheDir, `python-standalone-${process.platform}`);
  const marker = path.join(extractRoot, ".ready");

  if (!fs.existsSync(marker)) {
    await downloadFile(url, archive);
    fs.rmSync(extractRoot, { recursive: true, force: true });
    fs.mkdirSync(extractRoot, { recursive: true });
    execFileSync("tar", ["-xzf", archive, "-C", extractRoot], { stdio: "inherit" });
    fs.writeFileSync(marker, PBS_TAG);
  }

  const candidates =
    process.platform === "win32"
      ? [path.join(extractRoot, "python", "python.exe")]
      : [
          path.join(extractRoot, "python", "bin", "python3.12"),
          path.join(extractRoot, "python", "bin", "python3"),
        ];
  const py = candidates.find((p) => fs.existsSync(p));
  if (!py) throw new Error(`Portable Python not found under ${extractRoot}`);
  console.log("[setup-torch-tts] Portable Python:", py);
  return py;
}

function venvPython() {
  return process.platform === "win32"
    ? path.join(torchSrc, ".venv", "Scripts", "python.exe")
    : path.join(torchSrc, ".venv", "bin", "python");
}

function runPip(py, args) {
  const r = spawnSync(py, ["-m", "pip", ...args], {
    cwd: torchSrc,
    stdio: "inherit",
    env: projectCacheEnv(),
  });
  if (r.status !== 0) throw new Error(`pip failed: ${args.join(" ")}`);
}

async function main() {
  const { accel: accelArg, force } = parseArgs(process.argv.slice(2));
  const accel = accelArg || detectAccel();
  console.log(`[setup-torch-tts] accel=${accel} platform=${process.platform}`);

  if (!fs.existsSync(path.join(torchSrc, "tts_server.py"))) {
    throw new Error("tts/torch/tts_server.py missing");
  }

  const existing = venvPython();
  if (fs.existsSync(existing) && !force) {
    console.log("[setup-torch-tts] Reusing", existing, "(pass --force to recreate)");
  } else {
    if (force && fs.existsSync(path.join(torchSrc, ".venv"))) {
      fs.rmSync(path.join(torchSrc, ".venv"), { recursive: true, force: true });
    }
    const portable = await ensurePortablePython();
    console.log("[setup-torch-tts] Creating .venv…");
    const create = spawnSync(portable, ["-m", "venv", path.join(torchSrc, ".venv")], {
      cwd: torchSrc,
      stdio: "inherit",
      env: projectCacheEnv(),
    });
    if (create.status !== 0) throw new Error("venv creation failed");
  }

  const py = venvPython();
  if (!fs.existsSync(py)) throw new Error(`venv python missing: ${py}`);

  runPip(py, ["install", "--upgrade", "pip"]);

  // Base deps first (qwen-tts pulls a generic torchaudio from PyPI). Then overwrite
  // torch+torchaudio from the accel index so we never keep a CUDA libcudart wheel
  // on ROCm/CPU setups.
  runPip(py, ["install", "-r", path.join(torchSrc, "requirements-base.txt")]);

  if (accel === "rocm" && process.platform === "win32") {
    runPip(py, ["install", "--force-reinstall", "--no-cache-dir", ...ROCM_WINDOWS_WHEELS]);
  } else {
    runPip(py, [
      "install",
      "--force-reinstall",
      "--no-cache-dir",
      "-r",
      path.join(torchSrc, `requirements-${accel}.txt`),
      "--index-url",
      TORCH_INDEX[accel],
    ]);
  }

  // Record intended accel for the Node spawn (ROCm env vars).
  fs.writeFileSync(
    path.join(root, "tts-accel.json"),
    JSON.stringify(
      { platform: process.platform, accel, preparedAt: new Date().toISOString() },
      null,
      2
    )
  );

  console.log("[setup-torch-tts] Verifying imports…");
  const verify = `
import torch
import torchaudio
print('torch', torch.__version__)
print('torchaudio', torchaudio.__version__)
print('cuda_available', torch.cuda.is_available())
print('hip', getattr(torch.version, 'hip', None))
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0))
from qwen_tts import Qwen3TTSModel
print('qwen_tts ok')
print('ok')
`.trim();
  execFileSync(py, ["-c", verify], {
    stdio: "inherit",
    cwd: torchSrc,
    env: projectCacheEnv(),
  });

  console.log("\n[setup-torch-tts] Done. Restart: bun run electron:dev");
  console.log(`[setup-torch-tts] Smoke TTS: ${py} tts/torch/tts_server.py`);
}

main().catch((err) => {
  console.error("[setup-torch-tts]", err?.message || err);
  process.exit(1);
});
