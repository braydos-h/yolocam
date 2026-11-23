#!/usr/bin/env bash
# Download the ONNX Runtime 1.31.1 shared library used by the load-dynamic ort crate.
# Keep the version compatible with ort's api-31 feature (see Cargo.toml).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${ONNXRUNTIME_VERSION:-1.31.1}"
DEST_DIR="${1:-$ROOT}"
FORCE=0
if [[ "${2:-}" == "--force" || "${1:-}" == "--force" ]]; then
  FORCE=1
fi
if [[ "${1:-}" == "--force" ]]; then
  DEST_DIR="$ROOT"
fi

case "$(uname -s)" in
  Linux*)   PACKAGE_DIR="linux-x64";   LIB_NAME="libonnxruntime.so" ;;
  Darwin*)  PACKAGE_DIR="osx-arm64";   LIB_NAME="libonnxruntime.dylib" ;;
  *)        echo "Unsupported OS. Use the Windows PowerShell script or install ONNX Runtime manually." >&2; exit 1 ;;
esac

DEST="$DEST_DIR/$LIB_NAME"
if [[ -f "$DEST" && "$FORCE" -ne 1 ]]; then
  echo "ONNX Runtime already exists: $DEST"
  echo "Pass --force to replace it."
  exit 0
fi

mkdir -p "$DEST_DIR"
TMPROOT="$(mktemp -d "${TMPDIR:-/tmp}/yolocam-ort.XXXXXX")"
trap 'rm -rf "$TMPROOT"' EXIT
PKG="$TMPROOT/pkg"
mkdir -p "$PKG"
URL="https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/$VERSION/microsoft.ml.onnxruntime.$VERSION.nupkg"

echo "Downloading ONNX Runtime $VERSION from NuGet..."
if command -v curl >/dev/null 2>&1; then
  curl -fsSL --retry 3 --connect-timeout 20 --max-time 300 -o "$TMPROOT/ort.nupkg" "$URL"
else
  wget -O "$TMPROOT/ort.nupkg" "$URL"
fi

if command -v unzip >/dev/null 2>&1; then
  unzip -q "$TMPROOT/ort.nupkg" -d "$PKG"
else
  echo "The 'unzip' utility is required." >&2
  exit 1
fi

SRC="$PKG/runtimes/$PACKAGE_DIR/native/$LIB_NAME"
if [[ ! -f "$SRC" ]]; then
  echo "Shared library not found inside package: $SRC" >&2
  exit 1
fi

cp -f "$SRC" "$DEST"
echo "Installed: $DEST"
echo "ONNX Runtime version: $VERSION"
echo
echo "The app searches the working directory, the executable directory, ORT_DYLIB_PATH,"
echo "or yolo.onnxruntime_path for the shared library."
