#!/usr/bin/env bash
# Copy the built Aura Converter.app from release/ into /Applications.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RELEASE="${ROOT}/release"
DEST="/Applications/Aura Converter.app"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "Este script só roda no macOS." >&2
  exit 1
fi

APP=""
# Prefer electron-builder output dirs (mac, mac-arm64, mac-x64), then any *.app under release/
for dir in "${RELEASE}/mac-arm64" "${RELEASE}/mac" "${RELEASE}/mac-x64"; do
  candidate="${dir}/Aura Converter.app"
  if [[ -d "$candidate" ]]; then
    APP="$candidate"
    break
  fi
done

if [[ -z "$APP" ]]; then
  APP="$(find "$RELEASE" -maxdepth 3 -type d -name "Aura Converter.app" 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "$APP" || ! -d "$APP" ]]; then
  echo "Não achei Aura Converter.app em ${RELEASE}/." >&2
  echo "Rode antes: bun run dist:mac" >&2
  exit 1
fi

echo "Origem:  ${APP}"
echo "Destino: ${DEST}"

# Quit running instance if open (ignore errors when not running)
osascript -e 'tell application "Aura Converter" to quit' 2>/dev/null || true
sleep 0.5

# Replace existing install atomically-ish via staging + ditto
STAGING="$(mktemp -d "/tmp/aura-converter-install.XXXXXX")"
trap 'rm -rf "$STAGING"' EXIT

ditto "$APP" "${STAGING}/Aura Converter.app"
rm -rf "$DEST"
ditto "${STAGING}/Aura Converter.app" "$DEST"

# Drop Gatekeeper quarantine from local build copies
xattr -dr com.apple.quarantine "$DEST" 2>/dev/null || true

echo "Instalado em ${DEST}"
open -R "$DEST"
