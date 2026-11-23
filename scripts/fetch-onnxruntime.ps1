#Requires -Version 5.1
<#
.SYNOPSIS
  Downloads the CPU ONNX Runtime 1.31.1 DLL used by the load-dynamic ort crate.

.PARAMETER Version
  ONNX Runtime version. Keep this compatible with the `ort` crate (api-31 in
  Cargo.toml). The default is 1.31.1.

.PARAMETER Destination
  Directory in which to place onnxruntime.dll. Defaults to the repository root.

.PARAMETER Force
  Overwrite an existing onnxruntime.dll.
#>
[CmdletBinding()]
param(
    [string]$Version = "1.31.1",
    [string]$Destination = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
if ([string]::IsNullOrWhiteSpace($Destination)) {
    $Destination = $root
}
New-Item -ItemType Directory -Force -Path $Destination | Out-Null
$dll = Join-Path $Destination "onnxruntime.dll"

if ((Test-Path $dll) -and -not $Force) {
    Write-Host "ONNX Runtime already exists: $dll"
    Write-Host "Use -Force to replace it."
    exit 0
}

if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) {
    throw "curl.exe is required. It ships with Windows 10/11."
}

$packageName = "microsoft.ml.onnxruntime"
$packageUrl = "https://api.nuget.org/v3-flatcontainer/$packageName/$Version/$packageName.$Version.nupkg"
$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("yolocam-ort-" + [guid]::NewGuid().ToString("N"))
$zip = Join-Path $tempRoot "onnxruntime.zip"
$pkg = Join-Path $tempRoot "pkg"

try {
    New-Item -ItemType Directory -Force -Path $pkg | Out-Null
    Write-Host "Downloading ONNX Runtime $Version from NuGet..."
    & curl.exe -fsSL --retry 3 --connect-timeout 20 --max-time 300 -o $zip $packageUrl 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "ONNX Runtime download failed with exit code $LASTEXITCODE"
    }

    Expand-Archive -LiteralPath $zip -DestinationPath $pkg -Force
    $native = Join-Path $pkg "runtimes\win-x64\native\onnxruntime.dll"
    if (-not (Test-Path $native)) {
        throw "onnxruntime.dll not found inside the NuGet package."
    }
    Copy-Item -Force $native $dll
    Write-Host "Installed: $dll"
    Write-Host "ONNX Runtime version: $Version"
    Write-Host ""
    Write-Host "The app looks for onnxruntime.dll in its working directory / next to the executable."
}
finally {
    if (Test-Path $tempRoot) {
        Remove-Item -Recurse -Force $tempRoot -ErrorAction SilentlyContinue
    }
}
