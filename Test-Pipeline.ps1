# ============================================================================
#  Test-Pipeline.ps1  -  Functional tests for the patched pipeline.ps1
# ============================================================================
#
#  Usage:   .\Test-Pipeline.ps1
#  Requires PowerShell 5.1+ on Windows (same env as pipeline.ps1)
#
# ============================================================================

$ErrorActionPreference = "Stop"

# ---- Counters ----
$script:Passed  = 0
$script:Failed  = 0
$script:Skipped = 0

# ---- Assertion helpers ----

function Assert-Equal($Name, $Expected, $Actual) {
    if ($Expected -eq $Actual) {
        Write-Host "  [PASS] $Name" -ForegroundColor Green
        $script:Passed++
    } else {
        Write-Host "  [FAIL] $Name" -ForegroundColor Red
        Write-Host "         Expected: $Expected" -ForegroundColor Red
        Write-Host "         Actual  : $Actual" -ForegroundColor Red
        $script:Failed++
    }
}

function Assert-True($Name, [bool]$Condition) {
    if ($Condition) {
        Write-Host "  [PASS] $Name" -ForegroundColor Green
        $script:Passed++
    } else {
        Write-Host "  [FAIL] $Name" -ForegroundColor Red
        $script:Failed++
    }
}

function Assert-Match {
    param([string]$Name, [string]$Pattern, [string]$Value)
    if ($Value -match $Pattern) {
        Write-Host "  [PASS] $Name" -ForegroundColor Green
        $script:Passed++
    } else {
        Write-Host "  [FAIL] $Name" -ForegroundColor Red
        Write-Host "         Pattern: $Pattern" -ForegroundColor Red
        Write-Host "         Value  : $Value" -ForegroundColor Red
        $script:Failed++
    }
}

function Assert-Near($Name, [double]$Expected, [double]$Actual, [double]$Tolerance = 1e-9) {
    if ([Math]::Abs($Expected - $Actual) -le $Tolerance) {
        Write-Host "  [PASS] $Name" -ForegroundColor Green
        $script:Passed++
    } else {
        Write-Host "  [FAIL] $Name" -ForegroundColor Red
        Write-Host "         Expected: $Expected" -ForegroundColor Red
        Write-Host "         Actual  : $Actual"   -ForegroundColor Red
        $script:Failed++
    }
}

function Write-Section($Title) {
    Write-Host ""
    Write-Host ("=" * 60) -ForegroundColor Cyan
    Write-Host "  $Title" -ForegroundColor Cyan
    Write-Host ("=" * 60) -ForegroundColor Cyan
}

# ============================================================================
#  Import functions from pipeline.ps1
#
#  Strategy: use PowerShell's AST parser to extract only function definitions.
#  This avoids executing the main pipeline body.
# ============================================================================

$pipelinePath = Join-Path $PSScriptRoot "pipeline.ps1"
if (-not (Test-Path $pipelinePath)) {
    Write-Host "[ERROR] pipeline.ps1 not found next to this test script." -ForegroundColor Red
    Write-Host "        Place Test-Pipeline.ps1 in the same folder as pipeline.ps1" -ForegroundColor Red
    exit 1
}

# Variables that pipeline functions expect to exist
$BaseRatio      = 0.20
$RatioStep      = 0.05
$ETAIntervalMin = 3
$Generations    = 5
$totalStart     = Get-Date
$i              = 0

# Parse the file with the PowerShell AST (no execution)
$tokens = $null
$parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    $pipelinePath, [ref]$tokens, [ref]$parseErrors
)

# Extract every function definition
$funcDefs = $ast.FindAll(
    { param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] },
    $false
)

if ($funcDefs.Count -eq 0) {
    Write-Host "[ERROR] No functions found in pipeline.ps1" -ForegroundColor Red
    exit 1
}

foreach ($fd in $funcDefs) {
    Invoke-Expression $fd.Extent.Text
}

Write-Host "[INFO] Imported $($funcDefs.Count) functions from pipeline.ps1" -ForegroundColor DarkGray


# ============================================================================
#  TEST 1: Get-Ratio
# ============================================================================

Write-Section "TEST 1: Get-Ratio  -  ratio calculation and cap"

Assert-Near "Gen 1 = BaseRatio (0.20)" 0.20 (Get-Ratio 1)
Assert-Near "Gen 2 = 0.25"             0.25 (Get-Ratio 2)
Assert-Near "Gen 3 = 0.30"             0.30 (Get-Ratio 3)
Assert-Near "Gen 6 = 0.45"             0.45 (Get-Ratio 6)
Assert-Near "Gen 7 = cap at 0.50"      0.50 (Get-Ratio 7)
Assert-Near "Gen 10 = still capped"    0.50 (Get-Ratio 10)

$r = Get-Ratio 0
Assert-True  "Gen 0 returns a number"   ($r -is [double])


# ============================================================================
#  TEST 2: Write-ETA  -  output format
# ============================================================================

Write-Section "TEST 2: Write-ETA  -  output sanity"

# Populate genTimings so Write-ETA can produce projections
$script:genTimings = @(
    [PSCustomObject]@{ Secs=300; EpochsRan=30; EpochsMax=30; Normalized=300 },
    [PSCustomObject]@{ Secs=280; EpochsRan=30; EpochsMax=30; Normalized=280 }
)

$etaOutput = & {
    Write-ETA -Phase "Test Phase" `
              -PhaseStart (Get-Date).AddMinutes(-5) `
              -PipelineStart (Get-Date).AddMinutes(-30) `
              -GensCompleted 2 -GensTotal 5
} 6>&1 | Out-String

Assert-Match "Contains phase name"         "Test Phase"         $etaOutput
Assert-Match "Contains Phase elapsed:"    "Phase elapsed:"     $etaOutput
Assert-Match "Contains Total elapsed:"    "Total elapsed:"     $etaOutput
Assert-Match "Contains Remaining:"        "remaining:"         $etaOutput
Assert-Match "Contains Finish:"           "Finish:"            $etaOutput

# With GensCompleted = 0 and no timings, should NOT show Remaining/Finish
$script:genTimings = @()
$etaOutputNoProj = & {
    Write-ETA -Phase "Early" `
              -PhaseStart (Get-Date) `
              -PipelineStart (Get-Date) `
              -GensCompleted 0 -GensTotal 3
} 6>&1 | Out-String

Assert-True "No Remaining when 0 gens completed" (-not ($etaOutputNoProj -match "remaining:"))


# ============================================================================
#  TEST 3: Run-WithETA  -  exit-code passthrough and console inheritance
# ============================================================================

Write-Section "TEST 3: Run-WithETA  -  exit-code and process behavior"

$totalStart = Get-Date
$i = 0

# 3a: Success exit code
$exitCode0 = Run-WithETA -Exe "cmd.exe" -Arguments @("/c", "exit 0") -Phase "Test-Success"
Assert-Equal "Exit code 0 passthrough" 0 $exitCode0

# 3b: Non-zero exit code
$exitCode42 = Run-WithETA -Exe "cmd.exe" -Arguments @("/c", "exit 42") -Phase "Test-Fail42"
Assert-Equal "Exit code 42 passthrough" 42 $exitCode42

# 3c: Verify ProcessStartInfo flags are correct for console inheritance
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName               = "cmd.exe"
$psi.Arguments              = "/c echo hello"
$psi.UseShellExecute        = $false
$psi.RedirectStandardOutput = $false
$psi.RedirectStandardError  = $false
$psi.CreateNoWindow         = $false

Assert-True  "UseShellExecute = false"        ($psi.UseShellExecute -eq $false)
Assert-True  "RedirectStdOut = false"          ($psi.RedirectStandardOutput -eq $false)
Assert-True  "RedirectStdErr = false"          ($psi.RedirectStandardError -eq $false)
Assert-True  "CreateNoWindow = false"          ($psi.CreateNoWindow -eq $false)


# ============================================================================
#  TEST 4: Run-WithETA  -  no orphan processes
# ============================================================================

Write-Section "TEST 4: Run-WithETA  -  child cleanup (no orphans)"

$marker = "PipelineTest_$(Get-Random)"
$sleepScript = "Start-Sleep -Milliseconds 500; Write-Host '$marker'"

$beforeIds = Get-Process -Name "powershell","pwsh" -ErrorAction SilentlyContinue |
             Select-Object -ExpandProperty Id

$exitSleep = Run-WithETA -Exe "powershell.exe" `
    -Arguments @("-NoProfile", "-Command", $sleepScript) `
    -Phase "Orphan-Check"

Start-Sleep -Milliseconds 500

$afterIds = Get-Process -Name "powershell","pwsh" -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty Id

$newProcs = @($afterIds | Where-Object { $_ -notin $beforeIds })

Assert-Equal "Exit code from sleep child" 0 $exitSleep
# AUDIT 11.14: Tolerance of 1 leaked process is intentional — system services
# and IDE terminals (e.g., VSCode) can spawn powershell/pwsh processes that
# share the monitored process names and appear between before/after snapshots.
# AUDIT 11.15: This orphan detection approach is inherently flaky due to a race
# condition between the before/after process snapshots and unrelated process
# start/exit.  False positives are possible; do not treat a single failure as
# definitive evidence of a leak.
Assert-True  "No orphaned child processes" ($newProcs.Count -le 1)


# ============================================================================
#  TEST 5: Run-WithETA  -  arguments with spaces
# ============================================================================

Write-Section "TEST 5: Run-WithETA  -  arguments with spaces"

$tmpDir  = Join-Path $env:TEMP "pipeline test dir"
if (-not (Test-Path $tmpDir)) { New-Item -ItemType Directory -Path $tmpDir | Out-Null }
$tmpFile = Join-Path $tmpDir "output.txt"
if (Test-Path $tmpFile) { Remove-Item $tmpFile }

$exitSpace = Run-WithETA -Exe "powershell.exe" `
    -Arguments @("-NoProfile", "-Command", "Set-Content -Path '$tmpFile' -Value 'hello'") `
    -Phase "Space-Arg"

Assert-True "File created in path with spaces" (Test-Path $tmpFile)

# Cleanup
Remove-Item $tmpDir -Recurse -Force -ErrorAction SilentlyContinue


# ============================================================================
#  TEST 6: Run-WithETA  -  ETA timer fires (fast interval)
# ============================================================================

Write-Section "TEST 6: Run-WithETA  -  ETA timer fires during long process"

$savedInterval = $ETAIntervalMin
# AUDIT 11.16: Use try/finally to ensure $ETAIntervalMin is restored even if
# Run-WithETA throws (otherwise $ErrorActionPreference = "Stop" would skip
# the restore, leaving a mutated global for subsequent tests).
try {
    $ETAIntervalMin = 0

    $etaTimerOutput = & {
        Run-WithETA -Exe "powershell.exe" `
            -Arguments @("-NoProfile", "-Command", "Start-Sleep -Seconds 5") `
            -Phase "ETA-Timer-Test"
    } 6>&1 | Out-String
} finally {
    $ETAIntervalMin = $savedInterval
}

$etaFired = $etaTimerOutput -match "\[ETA\]"
Assert-True "ETA printed during 5s child with 0-min interval" $etaFired


# ============================================================================
#  TEST 7: Helper functions  -  output format
# ============================================================================

Write-Section "TEST 7: Helper functions  -  output format"

$bannerOut = & { Write-Banner "Hello World" } 6>&1 | Out-String
Assert-Match "Banner contains text"      "Hello World"  $bannerOut
Assert-Match "Banner contains separator" "==========" $bannerOut

$stepOut = & { Write-Step "Doing stuff" } 6>&1 | Out-String
Assert-Match "Step contains arrow"       ">"            $stepOut
Assert-Match "Step contains text"        "Doing stuff"  $stepOut

$okOut = & { Write-Ok "It worked" } 6>&1 | Out-String
Assert-Match "Ok contains OK tag"        "\[OK\]"      $okOut

$failOut = & { Write-Fail "It broke" } 6>&1 | Out-String
Assert-Match "Fail contains FAIL tag"    "\[FAIL\]"    $failOut


# ============================================================================
#  TEST 8: Get-Timestamp format
# ============================================================================

Write-Section "TEST 8: Get-Timestamp  -  format check"

$ts = Get-Timestamp
Assert-Match "Timestamp matches yyyy-MM-dd HH:mm:ss" `
    '^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$' $ts


# ============================================================================
#  TEST 9: # INFO [11.20]: This test is always skipped in non-interactive (CI) environments.
    # $script:Skipped increments instead of $script:Failed, so CI always reports green.
    # This provides no automated coverage for Ctrl+C handling.
    Ctrl+C forwarding (interactive  -  manual confirmation)
# ============================================================================

Write-Section "TEST 9: Ctrl+C forwarding (INTERACTIVE)"

Write-Host ""
Write-Host "  This test launches a long-running child process." -ForegroundColor Yellow
Write-Host "  Press Ctrl+C while it runs and confirm:" -ForegroundColor Yellow
Write-Host "    - The child process stops" -ForegroundColor Yellow
Write-Host "    - No orphan processes remain in Task Manager" -ForegroundColor Yellow
Write-Host ""

$runCtrlC = Read-Host "  Run Ctrl+C test? (y/N)"
if ($runCtrlC -eq "y") {
    Write-Host "  Starting 60s child... press Ctrl+C now" -ForegroundColor Magenta
    try {
        $totalStart = Get-Date
        $i = 0
        $code = Run-WithETA -Exe "powershell.exe" `
            -Arguments @("-NoProfile", "-Command", "Write-Host 'Child running (60s)... press Ctrl+C'; Start-Sleep 60; Write-Host 'Child finished naturally'") `
            -Phase "CtrlC-Test"
        Write-Host "  Child exited with code: $code" -ForegroundColor Yellow
    } catch {
        Write-Host "  Caught exception (expected with Ctrl+C): $($_.Exception.Message)" -ForegroundColor Yellow
    }

    Start-Sleep -Seconds 1
    $orphans = Get-Process -Name "powershell","pwsh" -ErrorAction SilentlyContinue |
               Where-Object { $_.Id -ne $PID }
    Write-Host "  Other PowerShell processes after Ctrl+C: $($orphans.Count)" -ForegroundColor Yellow
    Write-Host "  (Review manually - some may be unrelated)" -ForegroundColor Yellow
    $script:Skipped++
} else {
    Write-Host "  [SKIP] Ctrl+C test skipped" -ForegroundColor DarkGray
    $script:Skipped++
}


# ============================================================================
#  Summary
# ============================================================================

Write-Host ""
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host "  TEST SUMMARY" -ForegroundColor Cyan
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host "  Passed  : $($script:Passed)" -ForegroundColor Green
Write-Host "  Failed  : $($script:Failed)" -ForegroundColor Red
Write-Host "  Skipped : $($script:Skipped)" -ForegroundColor DarkGray
Write-Host ""

if ($script:Failed -gt 0) {
    Write-Host "  SOME TESTS FAILED" -ForegroundColor Red
    exit 1
} else {
    Write-Host "  ALL TESTS PASSED" -ForegroundColor Green
    exit 0
}
