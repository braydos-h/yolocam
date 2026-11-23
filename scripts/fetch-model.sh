#!/usr/bin/env bash
# Download a known-good FP32 YOLO11n COCO ONNX model for a quick smoke test.
# For production, export your own model with Ultralytics (see README.md).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/models/yolo11n.onnx"
URL="https://huggingface.co/booknerd/yolov11/resolve/main/yolo11n.onnx"
EXPECTED_SHA256="f7770b7229d2865585c9deb639a93e6c1a1cab3cf548a433261b2983486df9da"

if [[ -f "$DEST" && "${1:-}" != "--force" ]]; then
  echo "Model already exists: $DEST"
  echo "Pass --force to replace it."
  exit 0
fi

mkdir -p "$ROOT/models"
TMP="$(mktemp "${TMPDIR:-/tmp}/yolocam-model.XXXXXX.onnx")"
trap 'rm -f "$TMP"' EXIT

echo "Downloading YOLO11n FP32 ONNX model..."
if command -v curl >/dev/null 2>&1; then
  curl -fsSL --retry 3 --connect-timeout 20 --max-time 300 -o "$TMP" "$URL"
else
  wget -O "$TMP" "$URL"
fi

if command -v sha256sum >/dev/null 2>&1; then
  ACTUAL="$(sha256sum "$TMP" | awk '{print $1}')"
elif command -v shasum >/dev/null 2>&1; then
  ACTUAL="$(shasum -a 256 "$TMP" | awk '{print $1}')"
else
  echo "No SHA-256 tool found (need sha256sum or shasum)." >&2
  exit 1
fi

if [[ "$ACTUAL" != "$EXPECTED_SHA256" ]]; then
  echo "SHA-256 mismatch. Expected $EXPECTED_SHA256, got $ACTUAL" >&2
  exit 1
fi

mv -f "$TMP" "$DEST"
trap - EXIT
echo "Saved: $DEST"
echo "SHA-256: $EXPECTED_SHA256"
echo
echo "Next: download ONNX Runtime with ./scripts/fetch-onnxruntime.sh"
