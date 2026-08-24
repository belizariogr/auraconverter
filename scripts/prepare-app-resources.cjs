/**
 * Assemble a self-contained runtime tree for Electron packaging:
 *   build/app-resources/
 *     dist/
 *     bin/ffmpeg (+ ffprobe)    (static binaries for M4B / convert)
 *     python/…                  (portable CPython 3.12; all OS, cached in build/cache)
 *     qwen3-tts-apple-silicon/  (darwin only)
 *     tts/torch/                (win32/linux only)
 *     tts-accel.json
 *
 * Voice previews are NOT bundled — generated at runtime under AURA_DATA_DIR.
 * Usage:
 *   node scripts/prepare-app-resources.cjs [--platform=darwin|win32|linux] [--accel=cuda|rocm|cpu]
 *
 * GPU wheels must be prepared on the target OS (no cross-compile of torch).
 */
const fs = require("fs");
const path = require("path");
const https = require("https");
const http = require("http");
const zlib = require("zlib");
const { execFileSync, spawnSync } = require("child_process");

const root = path.resolve(__dirname, "..");
const out = path.join(root, "build", "app-resources");
const qwenSrc = path.join(root, "qwen3-tts-apple-silicon");
const torchSrc = path.join(root, "tts", "torch");
const kokoroSrc = path.join(root, "tts", "kokoro");
const cacheDir = path.join(root, "build", "cache");

/**
 * Robust recursive delete — macOS often throws ENOTEMPTY on large trees
 * (Python.framework / site-packages) with a single fs.rmSync.
 */
function removeDirRobust(dir) {
  if (!fs.existsSync(dir)) return;

  // Move aside first so mkdir can proceed even if unlink is slow/locked.
  const trash = `${dir}.trash-${process.pid}-${Date.now()}`;
  try {
    fs.renameSync(dir, trash);
  } catch (err) {
    // Fallback: delete in place when rename fails (e.g. cross-device).
    try {
      fs.rmSync(dir, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
      return;
    } catch (err2) {
      if (process.platform !== "win32") {
        const r = spawnSync("rm", ["-rf", dir], { stdio: "ignore" });
        if (r.status === 0) return;
      }
      throw err2;
    }
  }

  const purge = (target) => {
    try {
      fs.rmSync(target, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
    } catch {
      if (process.platform !== "win32") {
        spawnSync("rm", ["-rf", target], { stdio: "ignore" });
      }
    }
  };

  // Best-effort sync purge; if it still fails, leave trash for next run / OS.
  purge(trash);
  // Clean leftover trash dirs from previous failed runs.
  try {
    const parent = path.dirname(dir);
    const base = path.basename(dir);
    for (const name of fs.readdirSync(parent)) {
      if (name.startsWith(`${base}.trash-`)) {
        purge(path.join(parent, name));
      }
    }
  } catch {
    // ignore
  }
}

/** Pinned portable CPython 3.12 (python-build-standalone). */
const PBS_TAG = "20260303";
const PBS_VERSION = "3.12.13";

/** Static ffmpeg/ffprobe from eugeneware/ffmpeg-static (includes both tools). */
const FFMPEG_STATIC_TAG = "b6.1.1";
const FFMPEG_STATIC_BASE = `https://github.com/eugeneware/ffmpeg-static/releases/download/${FFMPEG_STATIC_TAG}`;

/** AMD ROCm Windows wheels (Python 3.12 / ROCm 7.2). Update when AMD bumps releases. */
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
  const electronCache = path.join(cacheDir, "electron");
  fs.mkdirSync(pipCache, { recursive: true });
  fs.mkdirSync(tmp, { recursive: true });
  fs.mkdirSync(hf, { recursive: true });
  fs.mkdirSync(electronCache, { recursive: true });
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
    electron_config_cache: process.env.electron_config_cache || electronCache,
    ELECTRON_CACHE: process.env.ELECTRON_CACHE || electronCache,
  };
}

/**
 * `prepare` is also an npm/bun install lifecycle hook. Skip the heavy pack
 * during install/add/ci so `bun install` stays fast. Explicit
 * `bun run prepare` / `bun run prepare:mac` still runs (npm_command=run-script,
 * or no npm_command).
 */
function shouldSkipLifecyclePrepare(argv) {
  if (argv.some((a) => a.startsWith("--platform=") || a === "--pack")) {
    return false;
  }
  if (process.env.AURA_PREPARE === "1" || process.env.AURA_PREPARE === "true") {
    return false;
  }
  const cmd = process.env.npm_command;
  return ["install", "ci", "add", "update", "remove", "unlink"].includes(cmd);
}

function ensureAppBuild() {
  const server = path.join(root, "dist", "server.cjs");
  const html = path.join(root, "dist", "index.html");
  if (fs.existsSync(server) && fs.existsSync(html)) return;
  console.log("[prepare-app-resources] dist/ missing — running bun run build:app");
  const r = spawnSync("bun", ["run", "build:app"], {
    cwd: root,
    stdio: "inherit",
    env: projectCacheEnv(),
  });
  if (r.status !== 0) {
    throw new Error("[prepare-app-resources] bun run build:app failed");
  }
}

function parseArgs(argv) {
  let platform = process.platform;
  let accel = "cuda";
  for (const arg of argv) {
    if (arg.startsWith("--platform=")) platform = arg.slice("--platform=".length);
    else if (arg.startsWith("--accel=")) accel = arg.slice("--accel=".length);
  }
  if (!["darwin", "win32", "linux"].includes(platform)) {
    throw new Error(`Invalid --platform=${platform}`);
  }
  if (!["cuda", "rocm", "cpu", "mlx"].includes(accel)) {
    throw new Error(`Invalid --accel=${accel}`);
  }
  if (platform === "darwin") accel = "mlx";
  else if (accel === "mlx") {
    throw new Error("--accel=mlx is only valid with --platform=darwin");
  }
  return { platform, accel };
}

function mustExist(p, label) {
  if (!fs.existsSync(p)) {
    throw new Error(`[prepare-app-resources] Missing ${label}: ${p}`);
  }
}

function copyFiltered(src, dst, skipNames = new Set()) {
  fs.mkdirSync(dst, { recursive: true });
  for (const name of fs.readdirSync(src)) {
    if (skipNames.has(name)) continue;
    if (name === "__pycache__" || name === ".git" || name === ".DS_Store") continue;
    if (name.endsWith(".pyc")) continue;
    const from = path.join(src, name);
    const to = path.join(dst, name);
    const st = fs.lstatSync(from);
    if (st.isDirectory()) {
      copyFiltered(from, to, skipNames);
    } else if (st.isSymbolicLink()) {
      try {
        fs.copyFileSync(fs.realpathSync(from), to);
      } catch {
        fs.cpSync(from, to, { dereference: true });
      }
    } else {
      fs.copyFileSync(from, to);
    }
  }
}

function writeAccelMeta(platform, accel) {
  const meta = { platform, accel, preparedAt: new Date().toISOString() };
  fs.writeFileSync(path.join(out, "tts-accel.json"), JSON.stringify(meta, null, 2));
  console.log("[prepare-app-resources] Wrote tts-accel.json", meta);
}

function seedCommon() {
  mustExist(path.join(root, "dist", "server.cjs"), "dist/server.cjs");
  mustExist(path.join(root, "dist", "index.html"), "dist/index.html");
  fs.cpSync(path.join(root, "dist"), path.join(out, "dist"), { recursive: true });

  // Native / heavy packages externalized from server.cjs bundle
  const nmOut = path.join(out, "node_modules");
  const externalPkgs = ["@napi-rs/canvas", "pdfjs-dist"];
  for (const pkg of externalPkgs) {
    const src = path.join(root, "node_modules", pkg);
    mustExist(src, `node_modules/${pkg}`);
    const dst = path.join(nmOut, pkg);
    fs.mkdirSync(path.dirname(dst), { recursive: true });
    fs.cpSync(src, dst, { recursive: true });
    console.log(`[prepare-app-resources] Bundled node_modules/${pkg}`);
  }
  // Platform-specific canvas binaries (optionalDependencies)
  const canvasPlatform = fs
    .readdirSync(path.join(root, "node_modules", "@napi-rs"))
    .filter((n) => n.startsWith("canvas-"));
  for (const name of canvasPlatform) {
    const src = path.join(root, "node_modules", "@napi-rs", name);
    const dst = path.join(nmOut, "@napi-rs", name);
    fs.mkdirSync(path.dirname(dst), { recursive: true });
    fs.cpSync(src, dst, { recursive: true });
    console.log(`[prepare-app-resources] Bundled node_modules/@napi-rs/${name}`);
  }

  const envSrc = path.join(root, ".env");
  if (fs.existsSync(envSrc)) {
    fs.copyFileSync(envSrc, path.join(out, ".env"));
    console.log("[prepare-app-resources] Included .env seed");
  }

  const repairSrc = path.join(root, "textRepair", "repair.py");
  if (fs.existsSync(repairSrc)) {
    const repairDst = path.join(out, "textRepair");
    fs.mkdirSync(repairDst, { recursive: true });
    fs.copyFileSync(repairSrc, path.join(repairDst, "repair.py"));
    console.log("[prepare-app-resources] Bundled textRepair/repair.py");
  }
}

function mlxSitePackages() {
  const lib = path.join(qwenSrc, ".venv", "lib");
  if (!fs.existsSync(lib)) {
    return path.join(qwenSrc, ".venv", "lib", "python3.12", "site-packages");
  }
  const pyDirs = fs.readdirSync(lib).filter((n) => n.startsWith("python"));
  const preferred = pyDirs.find((n) => n.includes("3.12")) || pyDirs[0];
  if (!preferred) {
    return path.join(qwenSrc, ".venv", "lib", "python3.12", "site-packages");
  }
  return path.join(lib, preferred, "site-packages");
}

function ensureMlxVenv() {
  console.log("[prepare-app-resources] Ensuring qwen3-tts-apple-silicon/.venv via setup-mlx-tts…");
  const setup = spawnSync(
    process.execPath,
    [path.join(__dirname, "setup-mlx-tts.cjs")],
    { cwd: root, stdio: "inherit", env: projectCacheEnv() }
  );
  if (setup.status !== 0) {
    throw new Error("setup-mlx-tts.cjs failed while preparing MLX runtime");
  }
}

async function prepareDarwin() {
  mustExist(path.join(qwenSrc, "tts_server.py"), "tts_server.py");
  mustExist(path.join(kokoroSrc, "tts_server_mlx.py"), "tts/kokoro/tts_server_mlx.py");
  mustExist(path.join(kokoroSrc, "tts_server.py"), "tts/kokoro/tts_server.py");

  const { pythonPrefix } = await ensurePortablePython("darwin");
  ensureMlxVenv();
  ensureKokoroVenv("darwin", "cpu");

  const siteSrc = mlxSitePackages();
  mustExist(siteSrc, "venv site-packages");
  if (!fs.existsSync(path.join(siteSrc, "misaki"))) {
    throw new Error(
      "[prepare-app-resources] misaki missing from MLX site-packages after setup-mlx-tts"
    );
  }

  const pyBin = bundlePortablePython(pythonPrefix);
  console.log("[prepare-app-resources] Bundled python:", pyBin);

  const qwenDst = path.join(out, "qwen3-tts-apple-silicon");
  fs.mkdirSync(qwenDst, { recursive: true });
  for (const file of ["tts_server.py", "requirements.txt", "main.py", "README.md"]) {
    const src = path.join(qwenSrc, file);
    if (fs.existsSync(src)) fs.copyFileSync(src, path.join(qwenDst, file));
  }
  console.log("[prepare-app-resources] Skipping models/ (downloaded on first launch)");
  console.log("[prepare-app-resources] Copying MLX site-packages...");
  copyFiltered(siteSrc, path.join(qwenDst, "site-packages"));

  const kokoroOnnxSiteSrc = kokoroSitePackages("darwin");
  mustExist(kokoroOnnxSiteSrc, "tts/kokoro/.venv site-packages");
  const kokoroDst = path.join(out, "tts", "kokoro");
  fs.mkdirSync(kokoroDst, { recursive: true });
  for (const file of [
    "tts_server_mlx.py",
    "tts_server.py",
    "requirements.txt",
    "README.md",
  ]) {
    const src = path.join(kokoroSrc, file);
    if (fs.existsSync(src)) fs.copyFileSync(src, path.join(kokoroDst, file));
  }
  copyFiltered(kokoroOnnxSiteSrc, path.join(kokoroDst, "site-packages"));
  console.log("[prepare-app-resources] Bundled Kokoro MLX + ONNX/Core ML runtimes");

  const pythonHome = path.join(out, "python");
  try {
    execFileSync(
      pyBin,
      ["-c", "import fastapi, uvicorn, mlx, mlx_audio, misaki; print('ok', fastapi.__version__)"],
      {
        env: {
          ...projectCacheEnv(),
          PYTHONHOME: pythonHome,
          PYTHONPATH: path.join(qwenDst, "site-packages"),
          PYTHONNOUSERSITE: "1",
        },
        stdio: "inherit",
      }
    );
    execFileSync(
      pyBin,
      [
        "-c",
        "import fastapi, uvicorn, onnxruntime as ort, kokoro_onnx; print('ok', ort.get_available_providers())",
      ],
      {
        env: {
          ...projectCacheEnv(),
          PYTHONHOME: pythonHome,
          PYTHONPATH: path.join(kokoroDst, "site-packages"),
          PYTHONNOUSERSITE: "1",
        },
        stdio: "inherit",
      }
    );
  } catch (err) {
    throw new Error(
      "[prepare-app-resources] Bundled Python failed to import TTS deps. " +
        (err?.message || err)
    );
  }
}

function pbsAssetName(platform) {
  const arch = process.arch === "arm64" ? "aarch64" : "x86_64";
  if (platform === "win32") {
    return `cpython-${PBS_VERSION}+${PBS_TAG}-${arch}-pc-windows-msvc-install_only.tar.gz`;
  }
  if (platform === "darwin") {
    return `cpython-${PBS_VERSION}+${PBS_TAG}-${arch}-apple-darwin-install_only.tar.gz`;
  }
  if (platform === "linux") {
    return `cpython-${PBS_VERSION}+${PBS_TAG}-${arch}-unknown-linux-gnu-install_only.tar.gz`;
  }
  throw new Error(`No portable Python asset for ${platform}`);
}

function downloadFile(url, dest, { minBytes = 1_000_000 } = {}) {
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  if (fs.existsSync(dest) && fs.statSync(dest).size > minBytes) {
    console.log("[prepare-app-resources] Using cached", dest);
    return Promise.resolve();
  }
  console.log("[prepare-app-resources] Downloading", url);
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

function ffmpegStaticTriplet(platform) {
  const arch = process.arch;
  if (platform === "darwin") {
    if (arch === "arm64") return "darwin-arm64";
    if (arch === "x64") return "darwin-x64";
  } else if (platform === "win32") {
    return "win32-x64";
  } else if (platform === "linux") {
    if (arch === "arm64") return "linux-arm64";
    if (arch === "x64") return "linux-x64";
  }
  throw new Error(
    `[prepare-app-resources] No static ffmpeg for platform=${platform} arch=${arch}`
  );
}

async function bundleFfmpeg(platform) {
  const triplet = ffmpegStaticTriplet(platform);
  const binDir = path.join(out, "bin");
  fs.mkdirSync(binDir, { recursive: true });

  for (const tool of ["ffmpeg", "ffprobe"]) {
    const asset = `${tool}-${triplet}.gz`;
    const url = `${FFMPEG_STATIC_BASE}/${asset}`;
    const gzPath = path.join(cacheDir, "ffmpeg-static", asset);
    await downloadFile(url, gzPath, { minBytes: 500_000 });

    const destName = platform === "win32" ? `${tool}.exe` : tool;
    const dest = path.join(binDir, destName);
    const gunzipped = zlib.gunzipSync(fs.readFileSync(gzPath));
    fs.writeFileSync(dest, gunzipped, { mode: 0o755 });
    if (platform !== "win32") {
      fs.chmodSync(dest, 0o755);
    }
    // Drop macOS quarantine if present (downloaded binaries)
    if (platform === "darwin") {
      try {
        execFileSync("xattr", ["-dr", "com.apple.quarantine", dest], {
          stdio: "ignore",
        });
      } catch {
        // xattr may fail if attribute absent
      }
    }
    console.log(`[prepare-app-resources] Bundled bin/${destName}`);
  }
}

function extractTarGz(archive, destDir) {
  fs.mkdirSync(destDir, { recursive: true });
  execFileSync("tar", ["-xzf", archive, "-C", destDir], { stdio: "inherit" });
}

/**
 * Download + extract portable CPython 3.12 into build/cache/python-standalone/<platform>.
 * Returns path to python binary.
 */
async function ensurePortablePython(platform) {
  const asset = pbsAssetName(platform);
  const url = `https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_TAG}/${asset}`;
  const archive = path.join(cacheDir, asset);
  const extractRoot = path.join(cacheDir, `python-standalone-${platform}`);
  const marker = path.join(extractRoot, ".ready");

  if (!fs.existsSync(marker)) {
    await downloadFile(url, archive);
    fs.rmSync(extractRoot, { recursive: true, force: true });
    fs.mkdirSync(extractRoot, { recursive: true });
    extractTarGz(archive, extractRoot);
    fs.writeFileSync(marker, PBS_TAG);
  }

  // install_only layout: <extract>/python/bin/python3.12  (or python/python.exe)
  const candidates =
    platform === "win32"
      ? [
          path.join(extractRoot, "python", "python.exe"),
          path.join(extractRoot, "python.exe"),
        ]
      : [
          path.join(extractRoot, "python", "bin", "python3.12"),
          path.join(extractRoot, "python", "bin", "python3"),
          path.join(extractRoot, "bin", "python3.12"),
        ];
  const py = candidates.find((p) => fs.existsSync(p));
  if (!py) {
    throw new Error(
      `[prepare-app-resources] Portable Python binary not found under ${extractRoot}`
    );
  }
  console.log("[prepare-app-resources] Portable Python:", py);
  return { pythonBin: py, pythonPrefix: path.join(extractRoot, "python") };
}

function torchVenvPython(platform) {
  if (platform === "win32") {
    return path.join(torchSrc, ".venv", "Scripts", "python.exe");
  }
  return path.join(torchSrc, ".venv", "bin", "python");
}

function torchSitePackages(platform) {
  if (platform === "win32") {
    return path.join(torchSrc, ".venv", "Lib", "site-packages");
  }
  const lib = path.join(torchSrc, ".venv", "lib");
  if (!fs.existsSync(lib)) {
    return path.join(torchSrc, ".venv", "lib", "python3.12", "site-packages");
  }
  const pyDirs = fs.readdirSync(lib).filter((n) => n.startsWith("python"));
  const preferred = pyDirs.find((n) => n.includes("3.12")) || pyDirs[0];
  if (!preferred) {
    return path.join(torchSrc, ".venv", "lib", "python3.12", "site-packages");
  }
  return path.join(lib, preferred, "site-packages");
}

function runPip(pythonBin, args) {
  const result = spawnSync(pythonBin, ["-m", "pip", ...args], {
    cwd: torchSrc,
    stdio: "inherit",
    env: projectCacheEnv(),
  });
  if (result.status !== 0) {
    throw new Error(`pip failed: ${args.join(" ")}`);
  }
}

function ensureTorchVenv(platform, accel, portablePy) {
  const py = torchVenvPython(platform);
  if (fs.existsSync(py) && fs.existsSync(torchSitePackages(platform))) {
    // Reuse only if it is 3.12.x (torch/ROCm wheels).
    try {
      const ver = execFileSync(py, ["-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"], {
        encoding: "utf8",
      }).trim();
      if (ver === "3.12") {
        console.log("[prepare-app-resources] Reusing existing tts/torch/.venv (3.12)");
        return py;
      }
      console.log(
        `[prepare-app-resources] Existing venv is Python ${ver}; recreating with 3.12...`
      );
      fs.rmSync(path.join(torchSrc, ".venv"), { recursive: true, force: true });
    } catch {
      fs.rmSync(path.join(torchSrc, ".venv"), { recursive: true, force: true });
    }
  }

  console.log(
    `[prepare-app-resources] Creating tts/torch/.venv (platform=${platform}, accel=${accel})...`
  );
  const create = spawnSync(
    portablePy,
    ["-m", "venv", path.join(torchSrc, ".venv")],
    { cwd: torchSrc, stdio: "inherit" }
  );
  if (create.status !== 0) {
    throw new Error("Failed to create tts/torch/.venv with portable Python 3.12.");
  }

  const venvPy = torchVenvPython(platform);
  mustExist(venvPy, "new venv python");

  const baseReq = path.join(torchSrc, "requirements-base.txt");
  mustExist(baseReq, "requirements-base.txt");
  runPip(venvPy, ["install", "--upgrade", "pip"]);
  // qwen-tts pulls a generic torchaudio from PyPI; overwrite torch+torchaudio
  // from the accel index afterward so ROCm/CPU builds never keep CUDA wheels.
  runPip(venvPy, ["install", "-r", baseReq]);

  if (accel === "rocm" && platform === "win32") {
    runPip(venvPy, ["install", "--force-reinstall", "--no-cache-dir", ...ROCM_WINDOWS_WHEELS]);
  } else {
    const accelReq = path.join(torchSrc, `requirements-${accel}.txt`);
    mustExist(accelReq, `requirements-${accel}.txt`);
    const index = TORCH_INDEX[accel];
    runPip(venvPy, [
      "install",
      "--force-reinstall",
      "--no-cache-dir",
      "-r",
      accelReq,
      "--index-url",
      index,
    ]);
  }

  return venvPy;
}

function bundlePortablePython(pythonPrefix) {
  const pythonDst = path.join(out, "python");
  console.log("[prepare-app-resources] Bundling portable Python from", pythonPrefix);
  fs.cpSync(pythonPrefix, pythonDst, { recursive: true, dereference: true });

  const candidates = [
    path.join(pythonDst, "python.exe"),
    path.join(pythonDst, "bin", "python3.12"),
    path.join(pythonDst, "bin", "python3"),
    path.join(pythonDst, "bin", "python"),
  ];
  const pyBin = candidates.find((p) => fs.existsSync(p));
  if (!pyBin) {
    throw new Error("[prepare-app-resources] No python binary after bundle");
  }
  return pyBin;
}

function kokoroVenvPython(platform) {
  return platform === "win32"
    ? path.join(kokoroSrc, ".venv", "Scripts", "python.exe")
    : path.join(kokoroSrc, ".venv", "bin", "python");
}

function kokoroSitePackages(platform) {
  const lib = path.join(kokoroSrc, ".venv", "lib");
  if (platform === "win32") {
    return path.join(kokoroSrc, ".venv", "Lib", "site-packages");
  }
  if (!fs.existsSync(lib)) return path.join(lib, "python3.12", "site-packages");
  const preferred = fs
    .readdirSync(lib)
    .find((n) => n.startsWith("python3."));
  if (!preferred) {
    return path.join(kokoroSrc, ".venv", "lib", "python3.12", "site-packages");
  }
  return path.join(lib, preferred, "site-packages");
}

function ensureKokoroVenv(platform, accel = "cpu") {
  const kokoroAccel =
    accel === "rocm" ? "rocm" : accel === "cuda" ? "cuda" : "cpu";
  // Prefer the dedicated setup script so ORT GPU wheels stay in one place.
  console.log(
    `[prepare-app-resources] Ensuring tts/kokoro/.venv (accel=${kokoroAccel}) via setup-kokoro-tts…`
  );
  const setup = spawnSync(
    process.execPath,
    [path.join(__dirname, "setup-kokoro-tts.cjs"), `--accel=${kokoroAccel}`],
    { cwd: root, stdio: "inherit", env: projectCacheEnv() }
  );
  if (setup.status !== 0) {
    throw new Error("setup-kokoro-tts.cjs failed while preparing Kokoro runtime");
  }
  const venvPy = kokoroVenvPython(platform);
  mustExist(venvPy, "kokoro venv python");
  return venvPy;
}

function bundleKokoroRuntime(platform) {
  mustExist(path.join(kokoroSrc, "tts_server.py"), "tts/kokoro/tts_server.py");
  const siteSrc = kokoroSitePackages(platform);
  mustExist(siteSrc, "tts/kokoro/.venv site-packages");

  const kokoroDst = path.join(out, "tts", "kokoro");
  fs.mkdirSync(kokoroDst, { recursive: true });
  for (const file of ["tts_server.py", "requirements.txt", "README.md"]) {
    const src = path.join(kokoroSrc, file);
    if (fs.existsSync(src)) fs.copyFileSync(src, path.join(kokoroDst, file));
  }
  console.log("[prepare-app-resources] Copying kokoro site-packages...");
  copyFiltered(siteSrc, path.join(kokoroDst, "site-packages"));
}

async function prepareWinLinux(platform, accel) {
  if (platform !== process.platform) {
    throw new Error(
      `[prepare-app-resources] Cannot prepare ${platform} torch wheels on ${process.platform}. ` +
        `Run this script on the target OS (or CI runner).`
    );
  }

  mustExist(path.join(torchSrc, "tts_server.py"), "tts/torch/tts_server.py");

  const { pythonBin: portablePy, pythonPrefix } = await ensurePortablePython(platform);
  const venvPy = ensureTorchVenv(platform, accel, portablePy);
  const siteSrc = torchSitePackages(platform);
  mustExist(siteSrc, "tts/torch/.venv site-packages");
  ensureKokoroVenv(platform, accel);
  const pyBin = bundlePortablePython(pythonPrefix);
  console.log("[prepare-app-resources] Bundled python:", pyBin);

  const torchDst = path.join(out, "tts", "torch");
  fs.mkdirSync(torchDst, { recursive: true });
  for (const file of [
    "tts_server.py",
    "requirements.txt",
    "requirements-base.txt",
    "requirements-cuda.txt",
    "requirements-rocm.txt",
    "requirements-cpu.txt",
    "README.md",
  ]) {
    const src = path.join(torchSrc, file);
    if (fs.existsSync(src)) fs.copyFileSync(src, path.join(torchDst, file));
  }
  console.log("[prepare-app-resources] Skipping models/ (downloaded on first launch)");
  console.log("[prepare-app-resources] Copying torch site-packages...");
  copyFiltered(siteSrc, path.join(torchDst, "site-packages"));
  bundleKokoroRuntime(platform);

  const pythonHome = path.join(out, "python");
  try {
    execFileSync(
      pyBin,
      [
        "-c",
        "import fastapi, uvicorn, torch; print('ok', torch.__version__, 'cuda', torch.cuda.is_available())",
      ],
      {
        env: {
          ...process.env,
          PYTHONHOME: pythonHome,
          PYTHONPATH: path.join(torchDst, "site-packages"),
          PYTHONNOUSERSITE: "1",
        },
        stdio: "inherit",
      }
    );
    execFileSync(
      pyBin,
      ["-c", "import fastapi, onnxruntime, kokoro_onnx; print('kokoro ok')"],
      {
        env: {
          ...process.env,
          PYTHONHOME: pythonHome,
          PYTHONPATH: path.join(out, "tts", "kokoro", "site-packages"),
          PYTHONNOUSERSITE: "1",
        },
        stdio: "inherit",
      }
    );
  } catch (err) {
    throw new Error(
      "[prepare-app-resources] Bundled Python failed to import TTS deps. " +
        (err?.message || err)
    );
  }
}

async function main() {
  const argv = process.argv.slice(2);
  if (shouldSkipLifecyclePrepare(argv)) {
    console.log(
      "[prepare-app-resources] Skipping pack during package install. Run: bun run prepare"
    );
    return;
  }

  const { platform, accel } = parseArgs(argv);
  console.log(
    `[prepare-app-resources] Preparing ${out} (platform=${platform}, accel=${accel})`
  );
  ensureAppBuild();
  removeDirRobust(out);
  fs.mkdirSync(out, { recursive: true });

  seedCommon();
  writeAccelMeta(platform, accel);
  await bundleFfmpeg(platform);

  if (platform === "darwin") {
    await prepareDarwin();
  } else {
    await prepareWinLinux(platform, accel);
  }

  let size = "?";
  try {
    size = execFileSync("du", ["-sh", out], { encoding: "utf8" }).trim();
  } catch {
    // du may be missing on Windows
  }
  console.log("[prepare-app-resources] Done:", size);
}

main().catch((err) => {
  console.error("[prepare-app-resources]", err?.message || err);
  process.exit(1);
});
