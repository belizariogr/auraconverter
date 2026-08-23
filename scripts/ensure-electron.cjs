/**
 * Ensures the Electron binary is present for the current platform.
 * bun/@electron get sometimes leaves a partial extract; fall back to unzip from cache.
 */
const { execFileSync } = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");

const root = path.resolve(__dirname, "..");
const projectElectronCache = path.join(root, "build", "cache", "electron");
fs.mkdirSync(projectElectronCache, { recursive: true });
if (!process.env.electron_config_cache) {
  process.env.electron_config_cache = projectElectronCache;
}
if (!process.env.ELECTRON_CACHE) {
  process.env.ELECTRON_CACHE = projectElectronCache;
}

const electronDir = path.join(root, "node_modules", "electron");
const distDir = path.join(electronDir, "dist");
const pathTxt = path.join(electronDir, "path.txt");

const platform = process.platform; // darwin | linux | win32
const arch = process.arch; // arm64 | x64 | ia32

function platformBinaryRel() {
  if (platform === "darwin") return "Electron.app/Contents/MacOS/Electron";
  if (platform === "win32") return "electron.exe";
  return "electron";
}

function isOk() {
  const binary = path.join(distDir, platformBinaryRel());
  if (!fs.existsSync(binary)) return false;
  if (platform === "darwin") {
    const frameworks = path.join(distDir, "Electron.app", "Contents", "Frameworks");
    return fs.existsSync(frameworks);
  }
  return true;
}

function writePathTxt() {
  fs.writeFileSync(pathTxt, platformBinaryRel());
}

function cacheRoots() {
  const roots = [];
  if (process.env.electron_config_cache) {
    roots.push(process.env.electron_config_cache);
  }
  if (platform === "darwin") {
    roots.push(path.join(os.homedir(), "Library", "Caches", "electron"));
  } else if (platform === "win32") {
    roots.push(
      path.join(
        process.env.LOCALAPPDATA || path.join(os.homedir(), "AppData", "Local"),
        "electron",
        "Cache"
      )
    );
  } else {
    roots.push(path.join(os.homedir(), ".cache", "electron"));
  }
  return roots;
}

function zipArch() {
  // electron release arch names
  if (arch === "arm64") return "arm64";
  if (arch === "ia32") return "ia32";
  return "x64";
}

function zipPlatform() {
  if (platform === "darwin") return "darwin";
  if (platform === "win32") return "win32";
  return "linux";
}

function findZip(dir, zipName, depth = 0) {
  if (!fs.existsSync(dir) || depth > 4) return null;
  for (const name of fs.readdirSync(dir)) {
    const full = path.join(dir, name);
    let st;
    try {
      st = fs.statSync(full);
    } catch {
      continue;
    }
    if (st.isDirectory()) {
      const found = findZip(full, zipName, depth + 1);
      if (found) return found;
    } else if (name === zipName) {
      return full;
    }
  }
  return null;
}

function unzipToDist(zipPath) {
  fs.rmSync(distDir, { recursive: true, force: true });
  fs.mkdirSync(distDir, { recursive: true });
  if (platform === "win32") {
    // PowerShell Expand-Archive as fallback when unzip is missing
    try {
      execFileSync("unzip", ["-q", zipPath, "-d", distDir], { stdio: "inherit" });
    } catch {
      execFileSync(
        "powershell.exe",
        [
          "-NoProfile",
          "-Command",
          `Expand-Archive -Path '${zipPath.replace(/'/g, "''")}' -DestinationPath '${distDir.replace(/'/g, "''")}' -Force`,
        ],
        { stdio: "inherit" }
      );
    }
  } else {
    execFileSync("unzip", ["-q", zipPath, "-d", distDir], { stdio: "inherit" });
  }
}

if (!fs.existsSync(electronDir)) {
  console.error(
    "[ensure-electron] node_modules/electron missing. Run: bun install"
  );
  process.exit(1);
}

if (isOk()) {
  writePathTxt();
  process.exit(0);
}

console.log(
  `[ensure-electron] Electron binary missing for ${platform}-${arch} — repairing...`
);

try {
  execFileSync(process.execPath, [path.join(electronDir, "install.js")], {
    cwd: electronDir,
    stdio: "inherit",
    env: { ...process.env, force_no_cache: "true" },
  });
} catch {
  // fall through to manual unzip
}

if (isOk()) {
  writePathTxt();
  process.exit(0);
}

const { version } = require(path.join(electronDir, "package.json"));
const zipName = `electron-v${version}-${zipPlatform()}-${zipArch()}.zip`;

let zipPath = null;
for (const cacheRoot of cacheRoots()) {
  zipPath = findZip(cacheRoot, zipName);
  if (zipPath) break;
}

if (!zipPath) {
  console.error(
    `[ensure-electron] Could not find ${zipName} under:\n  - ${cacheRoots().join("\n  - ")}\n` +
      "Try: rm -rf node_modules/electron && bun install && node node_modules/electron/install.js"
  );
  process.exit(1);
}

console.log("[ensure-electron] Unzipping", zipPath);
unzipToDist(zipPath);
writePathTxt();

if (!isOk()) {
  console.error("[ensure-electron] Repair failed after unzip.");
  process.exit(1);
}

console.log("[ensure-electron] Electron repaired.");
