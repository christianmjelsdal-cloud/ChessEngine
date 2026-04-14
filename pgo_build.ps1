# PGO (Profile-Guided Optimization) build script for MSVC
# Typically gives 10-15% speedup for search-heavy code.
#
# Usage:
#   .\pgo_build.ps1 [Release|Debug]
#
# Requirements:
#   - Visual Studio 2019+ with C++ workload
#   - Run from Developer PowerShell or Developer Command Prompt

param(
    [string]$Config = "Release"
)

$ErrorActionPreference = "Stop"

$SolutionDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$OutputDir = Join-Path $SolutionDir "x64\$Config"
$ExeName = "ChessEngine.exe"
$ExePath = Join-Path $OutputDir $ExeName

Write-Host "=== PGO Build: Phase 1 — Instrumented Build ===" -ForegroundColor Cyan

# Step 1: Build with /GL (whole program optimization) and instrumentation
$InstrFlags = "/GL /O2 /DNDEBUG"
$LinkFlags = "/LTCG:PGI /PGD:$OutputDir\ChessEngine.pgd"

# Use MSBuild with PGO flags
msbuild ChessEngine.vcxproj /p:Configuration=$Config /p:Platform=x64 `
    /p:WholeProgramOptimization=true `
    /p:LinkTimeCodeGeneration=PGInstrument `
    /t:Rebuild

if ($LASTEXITCODE -ne 0) {
    Write-Error "Instrumented build failed!"
    exit 1
}

Write-Host "`n=== PGO Build: Phase 2 — Training Run ===" -ForegroundColor Cyan
Write-Host "Running self-play with instrumented binary to collect profile data..."

# Run a short training workload (100 games at depth 5 is enough for profiling)
& $ExePath --generate --games 100 --depth 5 --workers 1

if ($LASTEXITCODE -ne 0) {
    Write-Warning "Training run returned non-zero exit code, but profile data may still be usable."
}

# FIX 12.11: Validate that profile data was actually generated before proceeding
$pgcCheck = Get-ChildItem -Path $OutputDir -Filter "*.pgc" -ErrorAction SilentlyContinue
if (-not $pgcCheck) {
    Write-Error "No .pgc profile files found in $OutputDir — PGO optimization cannot proceed."
    Write-Error "The training run may have failed to generate profile data."
    exit 1
}
Write-Host "  Found $($pgcCheck.Count) .pgc profile file(s)" -ForegroundColor Green

Write-Host "`n=== PGO Build: Phase 3 — Optimized Build ===" -ForegroundColor Cyan

# Step 3: Rebuild using the collected profile data
msbuild ChessEngine.vcxproj /p:Configuration=$Config /p:Platform=x64 `
    /p:WholeProgramOptimization=true `
    /p:LinkTimeCodeGeneration=PGOptimize `
    /t:Rebuild

if ($LASTEXITCODE -ne 0) {
    Write-Error "PGO-optimized build failed!"
    exit 1
}

Write-Host "`n=== PGO Build Complete ===" -ForegroundColor Green
Write-Host "Optimized binary: $ExePath"
Write-Host "Expected speedup: 10-15% for search operations"

# Optional: Clean up PGC files
$pgcFiles = Get-ChildItem -Path $OutputDir -Filter "*.pgc" -ErrorAction SilentlyContinue
if ($pgcFiles) {
    Write-Host "`nProfile data files ($($pgcFiles.Count) .pgc files) kept in $OutputDir"
    Write-Host "Delete them manually if no longer needed."
}
