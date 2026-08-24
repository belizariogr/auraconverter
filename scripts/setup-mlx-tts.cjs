/**
 * Create qwen3-tts-apple-silicon/.venv with portable CPython 3.12 + MLX.
 * Python tarball, pip cache and temp files stay under build/cache/.
 *
 *   bun run setup:tts:mlx
 *   bun run setup:tts:mlx -- --force
 */
const fs = require("fs");
const path = require("path");
const https = require("https");
const http = require("http");
const { execFileSync, spawnSync } = require("child_process");

const root = path.resolve(__dirname, "..");
const mlxSrc = path.join(root, "qwen3-tts-apple-silicon");
const cacheDir = path.join(root, "build", "cache");

const PBS_TAG = "20260303";
const PBS_VERSION = "3.12.13";

function parseArgs(argv) {
  let force = false;
  for (const arg of argv) {
    if (arg === "--force") force = true;
  }
  return { force };
}

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

function downloadFile(url, dest) {
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  if (fs.existsSync(dest) && fs.statSync(dest).size > 1_000_000) {
    console.log("[setup-mlx-tts] Using cached", dest);
    return Promise.resolve();
  }
  console.log("[setup-mlx-tts] Downloading", url);
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
  if (process.platform !== "darwin") {
    throw new Error("setup-mlx-tts is macOS-only (Apple Silicon / MLX)");
  }
  return `cpython-${PBS_VERSION}+${PBS_TAG}-${arch}-apple-darwin-install_only.tar.gz`;
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

  const candidates = [
    path.join(extractRoot, "python", "bin", "python3.12"),
    path.join(extractRoot, "python", "bin", "python3"),
  ];
  const py = candidates.find((p) => fs.existsSync(p));
  if (!py) throw new Error(`Portable Python not found under ${extractRoot}`);
  console.log("[setup-mlx-tts] Portable Python:", py);
  return py;
}

function venvPython() {
  return path.join(mlxSrc, ".venv", "bin", "python");
}

function pythonMinor(py) {
  try {
    return execFileSync(
      py,
      ["-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
      { encoding: "utf8" }
    ).trim();
  } catch {
    return "";
  }
}

function runPip(py, args) {
  const r = spawnSync(py, ["-m", "pip", ...args], {
    cwd: mlxSrc,
    stdio: "inherit",
    env: projectCacheEnv(),
  });
  if (r.status !== 0) throw new Error(`pip failed: ${args.join(" ")}`);
}

async function main() {
  if (process.platform !== "darwin") {
    throw new Error("setup-mlx-tts is only for macOS");
  }

  const { force } = parseArgs(process.argv.slice(2));
  console.log("[setup-mlx-tts] platform=darwin accel=mlx");

  if (!fs.existsSync(path.join(mlxSrc, "tts_server.py"))) {
    throw new Error("qwen3-tts-apple-silicon/tts_server.py missing");
  }
  if (!fs.existsSync(path.join(mlxSrc, "requirements.txt"))) {
    throw new Error("qwen3-tts-apple-silicon/requirements.txt missing");
  }

  const existing = venvPython();
  const existingOk = fs.existsSync(existing) && pythonMinor(existing) === "3.12";
  if (existingOk && !force) {
    console.log("[setup-mlx-tts] Reusing", existing, "(pass --force to recreate)");
  } else {
    if (fs.existsSync(path.join(mlxSrc, ".venv"))) {
      fs.rmSync(path.join(mlxSrc, ".venv"), { recursive: true, force: true });
    }
    const portable = await ensurePortablePython();
    console.log("[setup-mlx-tts] Creating .venv with portable CPython 3.12…");
    const create = spawnSync(portable, ["-m", "venv", path.join(mlxSrc, ".venv")], {
      cwd: mlxSrc,
      stdio: "inherit",
      env: projectCacheEnv(),
    });
    if (create.status !== 0) throw new Error("venv creation failed");
  }

  const py = venvPython();
  if (!fs.existsSync(py)) throw new Error(`venv python missing: ${py}`);

  runPip(py, ["install", "--upgrade", "pip"]);
  runPip(py, ["install", "-r", path.join(mlxSrc, "requirements.txt")]);

  console.log("[setup-mlx-tts] Verifying imports…");
  execFileSync(
    py,
    ["-c", "import fastapi, uvicorn, mlx, mlx_audio, misaki; print('ok mlx', fastapi.__version__)"],
    { stdio: "inherit", cwd: mlxSrc, env: projectCacheEnv() }
  );

  console.log("\n[setup-mlx-tts] Done. Python lives under qwen3-tts-apple-silicon/.venv");
  console.log(`[setup-mlx-tts] Smoke: ${py} qwen3-tts-apple-silicon/tts_server.py`);
}

main().catch((err) => {
  console.error("[setup-mlx-tts]", err?.message || err);
  process.exit(1);
});
