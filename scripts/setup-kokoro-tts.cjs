/**
 * Create tts/kokoro/.venv with portable CPython 3.12 + kokoro-onnx.
 *
 *   bun run setup:tts:kokoro
 *   bun run setup:tts:kokoro -- --force
 *   bun run setup:tts:kokoro -- --accel=rocm   # AMD MIGraphX
 *   bun run setup:tts:kokoro -- --accel=cuda   # NVIDIA
 *   bun run setup:tts:kokoro -- --accel=cpu
 *   bun run setup:tts:kokoro -- --accel=auto   # default: detect GPU
 */
const fs = require("fs");
const path = require("path");
const https = require("https");
const http = require("http");
const { execFileSync, spawnSync } = require("child_process");

const root = path.resolve(__dirname, "..");
const kokoroSrc = path.join(root, "tts", "kokoro");
const cacheDir = path.join(root, "build", "cache");

const PBS_TAG = "20260303";
const PBS_VERSION = "3.12.13";

/** AMD hosts onnxruntime-migraphx wheels here (match installed ROCm major.minor). */
const ROCM_ORT_FIND_LINKS = [
  "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/",
  "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/",
  "https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.4/",
  "https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3.4/",
];

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
  let force = false;
  let accel = "auto";
  for (const arg of argv) {
    if (arg === "--force") force = true;
    else if (arg.startsWith("--accel=")) accel = arg.slice("--accel=".length).toLowerCase();
  }
  if (!["auto", "cpu", "cuda", "rocm", "dml"].includes(accel)) {
    throw new Error(`Invalid --accel=${accel} (use auto|cpu|cuda|rocm|dml)`);
  }
  return { force, accel };
}

function downloadFile(url, dest) {
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  if (fs.existsSync(dest) && fs.statSync(dest).size > 1_000_000) {
    console.log("[setup-kokoro-tts] Using cached", dest);
    return Promise.resolve();
  }
  console.log("[setup-kokoro-tts] Downloading", url);
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
  if (process.platform === "darwin") {
    return `cpython-${PBS_VERSION}+${PBS_TAG}-${arch}-apple-darwin-install_only.tar.gz`;
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
  console.log("[setup-kokoro-tts] Portable Python:", py);
  return py;
}

function venvPython() {
  return process.platform === "win32"
    ? path.join(kokoroSrc, ".venv", "Scripts", "python.exe")
    : path.join(kokoroSrc, ".venv", "bin", "python");
}

function runPip(py, args) {
  const r = spawnSync(py, ["-m", "pip", ...args], {
    cwd: kokoroSrc,
    stdio: "inherit",
    env: projectCacheEnv(),
  });
  if (r.status !== 0) throw new Error(`pip failed: ${args.join(" ")}`);
}

function detectAccel() {
  if (process.platform === "win32") {
    // DirectML works on Windows AMD/Intel without ROCm CUDA stacks.
    try {
      const out = execFileSync("wmic", ["path", "win32_VideoController", "get", "name"], {
        encoding: "utf8",
      });
      if (/nvidia/i.test(out)) return "cuda";
      if (/amd|radeon|intel/i.test(out)) return "dml";
    } catch {
      // ignore
    }
    return "cpu";
  }

  if (process.platform === "linux") {
    try {
      const out = execFileSync("lspci", ["-nn"], { encoding: "utf8" });
      if (/NVIDIA|10de:/i.test(out)) return "cuda";
      if (/AMD|ATI|Radeon|1002:/i.test(out)) return "rocm";
    } catch {
      // ignore
    }
  }

  return "cpu";
}

function resolveAccel(requested) {
  if (requested !== "auto") return requested;
  const detected = detectAccel();
  console.log(`[setup-kokoro-tts] auto-detect accel → ${detected}`);
  return detected;
}

function installOnnxRuntime(py, accel) {
  // Mutually exclusive ORT builds — uninstall all flavors first.
  runPip(py, [
    "uninstall",
    "-y",
    "onnxruntime",
    "onnxruntime-gpu",
    "onnxruntime-migraphx",
    "onnxruntime-directml",
    "onnxruntime-rocm",
  ]);

  if (accel === "cuda") {
    console.log("[setup-kokoro-tts] Installing onnxruntime-gpu (CUDA)…");
    runPip(py, ["install", "onnxruntime-gpu>=1.20.0"]);
    return;
  }

  if (accel === "dml") {
    console.log("[setup-kokoro-tts] Installing onnxruntime-directml…");
    runPip(py, ["install", "onnxruntime-directml>=1.20.0"]);
    return;
  }

  if (accel === "rocm") {
    if (process.platform !== "linux") {
      console.warn(
        "[setup-kokoro-tts] ROCm/MIGraphX wheels are Linux-only; falling back to CPU onnxruntime"
      );
      runPip(py, ["install", "onnxruntime>=1.20.0"]);
      return;
    }
    console.log(
      "[setup-kokoro-tts] Installing onnxruntime-migraphx (AMD ROCm). Requires system package: migraphx"
    );
    let lastErr = null;
    for (const findLinks of ROCM_ORT_FIND_LINKS) {
      try {
        console.log(`[setup-kokoro-tts] Trying ${findLinks}`);
        runPip(py, [
          "install",
          "onnxruntime-migraphx",
          "-f",
          findLinks,
        ]);
        // Older AMD notes asked for numpy 1.26; only pin if the wheel refuses numpy 2.
        return;
      } catch (err) {
        lastErr = err;
        console.warn(`[setup-kokoro-tts] Failed with ${findLinks}: ${err.message}`);
      }
    }
    console.warn(
      "[setup-kokoro-tts] MIGraphX wheel install failed — installing CPU onnxruntime. " +
        "Install system `migraphx` and re-run with --accel=rocm --force."
    );
    if (lastErr) console.warn(String(lastErr.message || lastErr));
    runPip(py, ["install", "onnxruntime>=1.20.0"]);
    return;
  }

  console.log("[setup-kokoro-tts] Installing CPU onnxruntime…");
  runPip(py, ["install", "onnxruntime>=1.20.0"]);
}

function writeAccelMarker(accel) {
  const marker = path.join(kokoroSrc, ".venv", "aura-kokoro-accel.json");
  fs.writeFileSync(
    marker,
    JSON.stringify({ accel, platform: process.platform, at: new Date().toISOString() }, null, 2)
  );
}

async function main() {
  const { force, accel: accelArg } = parseArgs(process.argv.slice(2));
  const accel = resolveAccel(accelArg);
  console.log(`[setup-kokoro-tts] platform=${process.platform} accel=${accel}`);

  if (!fs.existsSync(path.join(kokoroSrc, "tts_server.py"))) {
    throw new Error("tts/kokoro/tts_server.py missing");
  }

  const existing = venvPython();
  if (fs.existsSync(existing) && !force) {
    console.log("[setup-kokoro-tts] Reusing", existing, "(pass --force to recreate)");
  } else {
    if (force && fs.existsSync(path.join(kokoroSrc, ".venv"))) {
      fs.rmSync(path.join(kokoroSrc, ".venv"), { recursive: true, force: true });
    }
    const portable = await ensurePortablePython();
    console.log("[setup-kokoro-tts] Creating .venv…");
    const create = spawnSync(portable, ["-m", "venv", path.join(kokoroSrc, ".venv")], {
      cwd: kokoroSrc,
      stdio: "inherit",
      env: projectCacheEnv(),
    });
    if (create.status !== 0) throw new Error("venv creation failed");
  }

  const py = venvPython();
  if (!fs.existsSync(py)) throw new Error(`venv python missing: ${py}`);

  runPip(py, ["install", "--upgrade", "pip"]);
  // Base deps without onnxruntime (installed per-accel below).
  runPip(py, ["install", "-r", path.join(kokoroSrc, "requirements.txt")]);
  installOnnxRuntime(py, accel);
  writeAccelMarker(accel);

  console.log("[setup-kokoro-tts] Verifying imports…");
  execFileSync(
    py,
    [
      "-c",
      "import fastapi, uvicorn, onnxruntime as ort; import kokoro_onnx; "
      + "print('ok', ort.__version__, ort.get_available_providers())",
    ],
    { stdio: "inherit", cwd: kokoroSrc, env: projectCacheEnv() }
  );

  if (accel === "rocm") {
    console.log(
      "\n[setup-kokoro-tts] AMD tip: ensure system MIGraphX is installed, e.g.\n" +
        "  Arch:  sudo pacman -S migraphx\n" +
        "  Ubuntu/Radeon: follow AMD ROCm docs (migraphx + half).\n" +
        "RX 6000 (RDNA2) often needs HSA_OVERRIDE_GFX_VERSION=10.3.0 (Aura sets this)."
    );
  }

  console.log("\n[setup-kokoro-tts] Done. Select Kokoro in the app model setup, then download assets.");
  console.log(`[setup-kokoro-tts] Smoke: ${py} tts/kokoro/tts_server.py`);
}

main().catch((err) => {
  console.error("[setup-kokoro-tts]", err?.message || err);
  process.exit(1);
});
