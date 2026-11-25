#!/usr/bin/env bash
# YOLOCam installer for Linux and macOS.
#
# Installs the build prerequisites, fetches the YOLO model and ONNX Runtime,
# builds the release binary, and can optionally install a runnable bundle and a
# systemd unit.
#
# Run ./install.sh --help for the available options.

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$ROOT"

PREFIX=""
SYSTEM=0
SERVICE=0
START=0
INSTALL_DEPS=1
DO_BUILD=1
FORCE=0
CUDA=0
SERVICE_HINT=""

if [[ -t 1 ]]; then
    BLUE=$'\033[1;34m'
    GREEN=$'\033[1;32m'
    YELLOW=$'\033[1;33m'
    RED=$'\033[1;31m'
    RESET=$'\033[0m'
else
    BLUE=""
    GREEN=""
    YELLOW=""
    RED=""
    RESET=""
fi

log() { printf '%s==>%s %s\n' "$BLUE" "$RESET" "$*"; }
ok() { printf '%s==>%s %s\n' "$GREEN" "$RESET" "$*"; }
warn() { printf '%swarning:%s %s\n' "$YELLOW" "$RESET" "$*" >&2; }
die() {
    printf '%serror:%s %s\n' "$RED" "$RESET" "$*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
YOLOCam installer

Usage: ./install.sh [options]

Options:
  --prefix DIR    Install a runnable bundle (binary, model, ONNX Runtime,
                  config, detections and database) into DIR.
  --system        Install into /opt/yolocam (implies --prefix). Uses sudo when
                  not running as root.
  --service       Install a systemd unit for the installed bundle. Uses a user
                  unit unless --system is given.
  --start         Enable and start the systemd unit after installing it.
  --no-deps       Do not install missing OS packages (FFmpeg, compiler, curl,
                  unzip). The script fails if a prerequisite is still missing.
  --no-build      Skip `cargo build`; use target/release/yolocam if present.
  --force         Re-download the model and ONNX Runtime even if present.
  --cuda          Build with the CUDA execution provider feature. A
                  CUDA-enabled ONNX Runtime must be supplied separately.
  -h, --help      Show this help.

Without --prefix the script prepares the repository in place: it installs
prerequisites, fetches the model and runtime, and builds target/release/yolocam.

Examples:
  ./install.sh
  ./install.sh --prefix "$HOME/.local/share/yolocam"
  ./install.sh --system --service --start
EOF
}

while (($#)); do
    case "$1" in
        --prefix)
            [[ $# -ge 2 ]] || die "--prefix requires a directory"
            PREFIX="$2"
            shift 2
            ;;
        --prefix=*)
            PREFIX="${1#*=}"
            shift
            ;;
        --system)
            SYSTEM=1
            shift
            ;;
        --service)
            SERVICE=1
            shift
            ;;
        --start)
            START=1
            SERVICE=1
            shift
            ;;
        --no-deps)
            INSTALL_DEPS=0
            shift
            ;;
        --no-build)
            DO_BUILD=0
            shift
            ;;
        --force)
            FORCE=1
            shift
            ;;
        --cuda)
            CUDA=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1 (try --help)"
            ;;
    esac
done

if ((SYSTEM)); then
    if [[ -n "$PREFIX" && "$PREFIX" != "/opt/yolocam" ]]; then
        die "--system cannot be combined with a different --prefix"
    fi
    PREFIX="/opt/yolocam"
fi

if ((SERVICE)) && [[ -z "$PREFIX" ]]; then
    die "--service and --start require --prefix or --system"
fi

if [[ -n "$PREFIX" ]]; then
    case "$PREFIX" in
        *'"'* | *\\*)
            die "the install path must not contain double quotes or backslashes"
            ;;
    esac
fi

OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
    Linux)
        ORT_LIB="libonnxruntime.so"
        ;;
    Darwin)
        ORT_LIB="libonnxruntime.dylib"
        ;;
    *)
        die "unsupported operating system: $OS (use the Windows PowerShell scripts)"
        ;;
esac

have_compiler() {
    command -v cc >/dev/null 2>&1 ||
        command -v gcc >/dev/null 2>&1 ||
        command -v clang >/dev/null 2>&1
}

install_os_packages() {
    local description="$1"
    local sudo_cmd=()

    if ((EUID != 0)); then
        if command -v sudo >/dev/null 2>&1; then
            sudo_cmd=(sudo)
        else
            die "sudo is required to install $description. Install them manually or re-run as root."
        fi
    fi

    log "Installing missing prerequisites: $description"

    case "$OS" in
        Darwin)
            command -v brew >/dev/null 2>&1 ||
                die "Homebrew is required to install $description. Install it from https://brew.sh and re-run."
            brew install ffmpeg
            ;;
        Linux)
            if command -v apt-get >/dev/null 2>&1; then
                "${sudo_cmd[@]}" apt-get update
                "${sudo_cmd[@]}" apt-get install -y \
                    build-essential ffmpeg curl unzip ca-certificates
            elif command -v dnf >/dev/null 2>&1; then
                if ! "${sudo_cmd[@]}" dnf install -y \
                    gcc gcc-c++ make ffmpeg curl unzip; then
                    "${sudo_cmd[@]}" dnf install -y \
                        gcc gcc-c++ make ffmpeg-free curl unzip
                fi
            elif command -v yum >/dev/null 2>&1; then
                "${sudo_cmd[@]}" yum install -y \
                    gcc gcc-c++ make ffmpeg curl unzip
            elif command -v pacman >/dev/null 2>&1; then
                "${sudo_cmd[@]}" pacman -S --needed --noconfirm \
                    base-devel ffmpeg curl unzip
            elif command -v zypper >/dev/null 2>&1; then
                "${sudo_cmd[@]}" zypper --non-interactive install \
                    gcc gcc-c++ make ffmpeg curl unzip
            elif command -v apk >/dev/null 2>&1; then
                "${sudo_cmd[@]}" apk add --no-cache \
                    build-base ffmpeg curl unzip
            else
                die "no supported package manager found. Install $description manually."
            fi
            ;;
    esac
}

check_prerequisites() {
    local missing=()
    local names=()

    if ! command -v ffmpeg >/dev/null 2>&1; then
        missing+=(ffmpeg)
        names+=(FFmpeg)
    fi
    if ! command -v curl >/dev/null 2>&1 &&
        ! command -v wget >/dev/null 2>&1; then
        missing+=(curl)
        names+=(curl)
    fi
    if ! command -v unzip >/dev/null 2>&1; then
        missing+=(unzip)
        names+=(unzip)
    fi
    if ! have_compiler; then
        missing+=(cc)
        names+=(a C compiler)
    fi

    ((${#missing[@]} == 0)) && return 0

    local description
    description="$(printf '%s, ' "${names[@]}")"
    description="${description%, }"

    if ((INSTALL_DEPS)); then
        install_os_packages "$description"
    else
        die "missing prerequisites: $description (remove --no-deps or install them manually)"
    fi

    # Verify the packages actually provided what we need.
    command -v ffmpeg >/dev/null 2>&1 ||
        die "FFmpeg is still missing after package installation"
    if ! command -v curl >/dev/null 2>&1 &&
        ! command -v wget >/dev/null 2>&1; then
        die "curl or wget is still missing after package installation"
    fi
    command -v unzip >/dev/null 2>&1 ||
        die "unzip is still missing after package installation"
    have_compiler || die "a C compiler is still missing after package installation"
}

rust_version_ok() {
    local version="$1"
    local major minor rest

    major="${version%%.*}"
    rest="${version#*.}"
    minor="${rest%%[!0-9]*}"

    [[ "$major" =~ ^[0-9]+$ && "$minor" =~ ^[0-9]+$ ]] || return 1
    ((major > 1 || (major == 1 && minor >= 92)))
}

ensure_rust() {
    if [[ -f "$HOME/.cargo/env" ]]; then
        # shellcheck disable=SC1091
        source "$HOME/.cargo/env"
    fi
    export PATH="$HOME/.cargo/bin:$PATH"

    local version=""
    if command -v rustc >/dev/null 2>&1; then
        version="$(rustc --version 2>/dev/null | awk '{print $2}' || true)"
        if [[ -n "$version" ]] && rust_version_ok "$version"; then
            ok "Rust $version is available"
            return 0
        fi
        if [[ -n "$version" ]]; then
            warn "Rust $version is installed, but 1.92 or newer is required"
        else
            warn "rustc is installed but not usable; repairing the Rust toolchain"
        fi
    fi

    if command -v rustup >/dev/null 2>&1; then
        log "Installing the current stable Rust toolchain"
        rustup toolchain install stable --profile minimal
        rustup default stable
    else
        log "Installing Rust with rustup"
        if command -v curl >/dev/null 2>&1; then
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs |
                sh -s -- -y --profile minimal
        else
            wget -qO- https://sh.rustup.rs | sh -s -- -y --profile minimal
        fi
        # shellcheck disable=SC1091
        source "$HOME/.cargo/env"
    fi

    version="$(rustc --version 2>/dev/null | awk '{print $2}' || true)"
    if [[ -z "$version" ]] || ! rust_version_ok "$version"; then
        die "Rust ${version:-toolchain} is still older than the required 1.92"
    fi
    ok "Rust $version is available"
}

fetch_assets() {
    log "Fetching the YOLO11n ONNX model"
    if ((FORCE)); then
        bash "$ROOT/scripts/fetch-model.sh" --force
    else
        bash "$ROOT/scripts/fetch-model.sh"
    fi
    [[ -s "$ROOT/models/yolo11n.onnx" ]] ||
        die "models/yolo11n.onnx is missing after the download"

    case "$OS:$ARCH" in
        Linux:x86_64 | Darwin:arm64)
            log "Fetching ONNX Runtime 1.31.1"
            if ((FORCE)) || [[ ! -f "$ROOT/$ORT_LIB" ]]; then
                if ((FORCE)); then
                    bash "$ROOT/scripts/fetch-onnxruntime.sh" --force
                else
                    bash "$ROOT/scripts/fetch-onnxruntime.sh"
                fi
            else
                ok "ONNX Runtime already present: $ROOT/$ORT_LIB"
            fi
            ;;
        *)
            if [[ ! -f "$ROOT/$ORT_LIB" ]]; then
                die "automatic ONNX Runtime download supports x86_64 Linux and arm64 macOS only. Place $ORT_LIB in $ROOT manually (see README.md), then re-run."
            fi
            warn "using the existing $ORT_LIB (unsupported platform for automatic download)"
            ;;
    esac

    [[ -f "$ROOT/$ORT_LIB" ]] ||
        die "$ORT_LIB is missing after the download"
}

build_binary() {
    local build_args=(build --release --locked)
    if ((CUDA)); then
        build_args+=(--features cuda)
    fi

    if ((CUDA)); then
        warn "--cuda builds against the CUDA provider, but the downloaded ONNX Runtime is CPU-only"
        warn "replace libonnxruntime with a CUDA-enabled distribution before running"
    fi

    log "Building the release binary"
    cargo "${build_args[@]}"
}

install_bundle() {
    local sudo_cmd=()

    if ((EUID != 0)) && [[ ! -w "$(dirname "$PREFIX")" || -e "$PREFIX" && ! -w "$PREFIX" ]]; then
        if command -v sudo >/dev/null 2>&1; then
            sudo_cmd=(sudo)
        else
            die "cannot write to $PREFIX and sudo is not available; choose a writable --prefix or re-run as root"
        fi
    fi

    "${sudo_cmd[@]}" mkdir -p "$PREFIX"
    PREFIX="$(cd "$PREFIX" && pwd -P)"
    "${sudo_cmd[@]}" install -d -m 0755 \
        "$PREFIX/models" "$PREFIX/models/auto" "$PREFIX/detections"

    log "Installing the runtime bundle into $PREFIX"
    "${sudo_cmd[@]}" install -m 0755 "$ROOT/target/release/yolocam" "$PREFIX/yolocam"
    "${sudo_cmd[@]}" install -m 0755 "$ROOT/$ORT_LIB" "$PREFIX/$ORT_LIB"
    "${sudo_cmd[@]}" install -m 0644 \
        "$ROOT/models/yolo11n.onnx" "$PREFIX/models/yolo11n.onnx"

    if [[ -f "$ROOT/README.md" && -f "$ROOT/LICENSE" ]]; then
        "${sudo_cmd[@]}" install -m 0644 "$ROOT/README.md" "$ROOT/LICENSE" "$PREFIX/"
    fi

    if [[ ! -f "$PREFIX/config.toml" ]]; then
        local tmp model_config
        tmp="$(mktemp "${TMPDIR:-/tmp}/yolocam-config.XXXXXX")"

        # Keep hardware-based model selection ("auto") enabled when requested;
        # otherwise point at the bundled legacy model.
        model_config="$PREFIX/models/yolo11n.onnx"
        if grep -Eq '^[[:space:]]*model_path[[:space:]]*=[[:space:]]*"auto"' "$ROOT/config.toml"; then
            model_config="auto"
        fi

        MODEL_PATH="$model_config" \
            SCREENSHOT_DIR="$PREFIX/detections" \
            DATABASE_PATH="$PREFIX/yolocam.sqlite3" \
            ORT_PATH="$PREFIX/$ORT_LIB" \
            awk '
                BEGIN {
                    model = ENVIRON["MODEL_PATH"]
                    shots = ENVIRON["SCREENSHOT_DIR"]
                    db = ENVIRON["DATABASE_PATH"]
                    ort = ENVIRON["ORT_PATH"]
                    saw_yolo = 0
                }
                /^[[:space:]]*\[yolo\][[:space:]]*$/ {
                    print
                    print "onnxruntime_path = \"" ort "\""
                    saw_yolo = 1
                    next
                }
                /^[[:space:]]*#?[[:space:]]*onnxruntime_path[[:space:]]*=/ { next }
                /^[[:space:]]*model_path[[:space:]]*=/ {
                    print "model_path = \"" model "\""
                    next
                }
                /^[[:space:]]*screenshot_directory[[:space:]]*=/ {
                    print "screenshot_directory = \"" shots "\""
                    next
                }
                /^[[:space:]]*database_path[[:space:]]*=/ {
                    print "database_path = \"" db "\""
                    next
                }
                { print }
                END { if (!saw_yolo) exit 3 }
            ' "$ROOT/config.toml" >"$tmp" ||
            {
                rm -f "$tmp"
                die "could not generate config.toml: missing [yolo] section in config.toml"
            }

        "${sudo_cmd[@]}" install -m 0644 "$tmp" "$PREFIX/config.toml"
        rm -f "$tmp"
        ok "Created $PREFIX/config.toml"
    else
        ok "Keeping the existing config: $PREFIX/config.toml"
    fi
}

install_service() {
    local sudo_cmd=()
    local unit wanted enable_command

    if ! command -v systemctl >/dev/null 2>&1; then
        warn "systemd was not found; skipping service installation"
        return 0
    fi

    if ((SYSTEM)); then
        if ((EUID != 0)); then
            sudo_cmd=(sudo)
        fi
        unit="/etc/systemd/system/yolocam.service"
        wanted="multi-user.target"
        enable_command="sudo systemctl enable --now yolocam"

        "${sudo_cmd[@]}" install -d -m 0755 "$(dirname "$unit")"
        {
            cat <<EOF
[Unit]
Description=YOLOCam RTSP camera monitor
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory="$PREFIX"
ExecStart="$PREFIX/yolocam" --config "$PREFIX/config.toml"
Restart=on-failure
RestartSec=5
NoNewPrivileges=true
PrivateTmp=true
# Optional camera credentials:
# EnvironmentFile=-$PREFIX/yolocam.env

[Install]
WantedBy=$wanted
EOF
        } | "${sudo_cmd[@]}" tee "$unit" >/dev/null

        "${sudo_cmd[@]}" systemctl daemon-reload
    else
        unit="$HOME/.config/systemd/user/yolocam.service"
        wanted="default.target"
        enable_command="systemctl --user enable --now yolocam"

        install -d -m 0755 "$(dirname "$unit")"
        cat >"$unit" <<EOF
[Unit]
Description=YOLOCam RTSP camera monitor
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory="$PREFIX"
ExecStart="$PREFIX/yolocam" --config "$PREFIX/config.toml"
Restart=on-failure
RestartSec=5
NoNewPrivileges=true
PrivateTmp=true
# Optional camera credentials:
# EnvironmentFile=-$PREFIX/yolocam.env

[Install]
WantedBy=$wanted
EOF

        systemctl --user daemon-reload ||
            warn "could not reload the user systemd manager"
    fi

    ok "Installed $unit"

    if ((START)); then
        log "Starting yolocam"
        if ((SYSTEM)); then
            "${sudo_cmd[@]}" systemctl enable --now yolocam
        else
            systemctl --user enable --now yolocam ||
                warn "could not start the user service; is a user systemd session available?"
        fi
    else
        SERVICE_HINT="$enable_command"
    fi
}

check_prerequisites
ensure_rust
fetch_assets

if ((DO_BUILD)); then
    build_binary
fi

[[ -x "$ROOT/target/release/yolocam" ]] ||
    die "target/release/yolocam is missing (build it with ./install.sh or drop --no-build)"

if [[ -n "$PREFIX" ]]; then
    install_bundle
    if ((SERVICE)); then
        install_service
    fi
    CONFIG_PATH="$PREFIX/config.toml"
    printf -v RUN_COMMAND 'cd %q && ./yolocam --config config.toml' "$PREFIX"
else
    CONFIG_PATH="$ROOT/config.toml"
    printf -v RUN_COMMAND \
        'cd %q && ./target/release/yolocam --config config.toml' "$ROOT"
fi

if grep -q 'rtsp://192.168.1.100:554/stream1' "$CONFIG_PATH" 2>/dev/null; then
    warn "config still uses the placeholder RTSP URL: $CONFIG_PATH"
fi

PORT=""
if [[ -r "$CONFIG_PATH" ]]; then
    PORT="$(awk -F= '
        /^[[:space:]]*port[[:space:]]*=/ {
            value = $2
            sub(/#.*/, "", value)
            gsub(/[^0-9]/, "", value)
            print value
            exit
        }
    ' "$CONFIG_PATH" 2>/dev/null || true)"
fi
[[ -n "$PORT" ]] || PORT=8080

printf '\n'
ok "YOLOCam is ready"
cat <<EOF

  Config:  $CONFIG_PATH
  Run:     $RUN_COMMAND
  Open:    http://127.0.0.1:$PORT

Set camera.rtsp_url in the config, or export YOLOCAM_RTSP_URL before running.
For camera credentials, prefer YOLOCAM_RTSP_USERNAME / YOLOCAM_RTSP_PASSWORD
over storing them in the config file.
EOF

if [[ -n "$SERVICE_HINT" ]]; then
    cat <<EOF

The systemd unit was installed but not started. Enable it with:

  $SERVICE_HINT
EOF
fi
