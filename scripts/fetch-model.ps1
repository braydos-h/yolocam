#Requires -Version 5.1
<#
.SYNOPSIS
  Downloads a known-good FP32 YOLO11n COCO ONNX model for a quick smoke test.

.DESCRIPTION
  The model is a community mirror of an Ultralytics YOLO11n export. For
  production, export your own model with Ultralytics (see README.md). The
  download is hash-checked so a partially downloaded or altered file is not
  silently accepted.

.PARAMETER Force
  Overwrite an existing models/yolo11n.onnx.
#>
[CmdletBinding()]
param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$models = Join-Path $root "models"
$destination = Join-Path $models "yolo11n.onnx"
$url = "https://huggingface.co/booknerd/yolov11/resolve/main/yolo11n.onnx"
$expectedSha256 = "F7770B7229D2865585C9DEB639A93E6C1A1CAB3CF548A433261B2983486DF9DA"

if ((Test-Path $destination) -and -not $Force) {
    Write-Host "Model already exists: $destination"
    Write-Host "Use -Force to replace it."
    exit 0
}

if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) {
    throw "curl.exe is required. It ships with Windows 10/11."
}

New-Item -ItemType Directory -Force -Path $models | Out-Null
$temp = Join-Path ([System.IO.Path]::GetTempPath()) ("yolocam-model-" + [guid]::NewGuid() + ".onnx")

try {
    Write-Host "Downloading YOLO11n FP32 ONNX model..."
    & curl.exe -fsSL --retry 3 --connect-timeout 20 --max-time 300 -o $temp $url 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Model download failed with exit code $LASTEXITCODE"
    }

    $actual = (Get-FileHash -Path $temp -Algorithm SHA256).Hash.ToUpperInvariant()
    if ($actual -ne $expectedSha256) {
        throw "SHA-256 mismatch. Expected $expectedSha256, got $actual"
    }

    Move-Item -Force -Path $temp -Destination $destination
    Write-Host "Saved: $destination"
    Write-Host "SHA-256: $expectedSha256"
    Write-Host ""
    Write-Host "Next: download ONNX Runtime with scripts\fetch-onnxruntime.ps1"
}
finally {
    if (Test-Path $temp) {
        Remove-Item -Force $temp -ErrorAction SilentlyContinue
    }
}
