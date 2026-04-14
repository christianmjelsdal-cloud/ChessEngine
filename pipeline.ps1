# ============================================================================
#  NNUE Training Pipeline - Automated Multi-Generation Self-Play & Training
# ============================================================================
#
#  Usage:
#    .\pipeline.ps1 -StartGen 1 -Generations 1    # You're at Gen 1, produces Gen 2
#    .\pipeline.ps1 -StartGen 1 -Generations 3    # You're at Gen 1, produces Gen 2, 3, 4
#    .\pipeline.ps1 -StartGen 2 -Generations 4 -GamesPerGen 25000 -MaxPositions 800000 -Epochs 40
#
#  Plotting:
#    Default: training log cleared once at start, final plot shows all generations combined
#    -PlotPerGen: clears log before each gen, saves training_progress_genN.png per gen
#
#  Overfitting warnings are printed after each generation's training phase.
#  Severity levels:
#    [WARN]  val loss > train loss * 1.05 for 3+ consecutive epochs
#    [HIGH]  val loss > train loss * 1.10 for 5+ consecutive epochs
#    [INFO]  train loss falling but val loss flat/rising (diverging gap)
#    [CRIT]  loss explosion detected (>2x spike) or NaN/Inf in log
#
# ============================================================================

param(
    [Parameter(Mandatory=$true)]
    [int]$StartGen,

    [Parameter(Mandatory=$true)]
    [int]$Generations,

    # --- Self-Play ---
    [int]$GamesPerGen    = 5000,
    [int]$Depth          = 5,
    [int]$Workers        = 12,
    [string]$Openings    = "openings.txt",            # Path to opening book file (one FEN per line)

    # --- Training ---
    [int]$MaxPositions   = 300000,
    [int]$Epochs         = 20,
    [int]$EarlyStop      = 10,
    [int]$BatchSize      = 8192,
    [double]$LR          = 0.001,         # Recommended: 0.001 (Adam default, safe for long runs)
    [double]$LRMin       = 0.00001,       # LR floor for cosine decay (prevents grinding to a halt)
    [double]$WeightDecay = 0.0001,        # Recommended: 0.0001 (mild L2, primary anti-overfit lever)
    [double]$GradClip    = 1.0,           # Gradient clip norm (catches exploding gradients before weight corruption)
    [double]$Dropout     = 0.1,           # Dropout rate on hidden layers (0 to disable)
    [double]$LabelSmoothing = 0.0,        # Label smoothing on result targets (try 0.05 for extra regularisation)
    [double]$DrawWeight  = 1.0,
    [double]$MateBoost   = 3.0,           # Mate-boost multiplier for train_nnue.py (0 = disabled)

    # --- Warmup ---
    [int]$WarmupSteps    = -1,            # LR warmup optimizer steps (-1 = auto-calc ~3 epochs)

    # --- Self-Play Ratio ---
    [double]$BaseRatio   = 0.20,
    [double]$RatioStep   = 0.05,

    # --- Validation ---
    [switch]$ValidateElo,                 # Run quick match vs previous gen after training
    [int]$ValidationGames = 50,           # Games for validation match

    # --- Plotting ---
    [switch]$PlotPerGen,          # If set: clear log + save plot per generation
                                  # Default: clear log once at start, cumulative plot

    # --- Paths (defaults assume running from project root) ---
    [string]$Engine      = ".\x64\Release\ChessEngine.exe",
    [string]$AssetsDir   = "assets",
    [string]$TrainingData = "assets\training_data.bin",

    # --- Training internals ---
    [int]$GradAccum          = 4,         # Gradient accumulation steps (must match train_nnue.py --grad-accum)

    # --- ETA ---
    [int]$ETAIntervalMin     = 3,         # Print ETA every N minutes during all phases

    # --- Python ---  (F6.3: configurable Python version)
    [string]$PythonVersion   = "3.10"     # Python version for py launcher (e.g. "3.10", "3.11")
)

$ErrorActionPreference = "Stop"

# ============================================================================
#  Helpers
# ============================================================================

function Write-Banner($text) {
    $line = "=" * 70
    Write-Host ""
    Write-Host $line -ForegroundColor Cyan
    Write-Host "  $text" -ForegroundColor Cyan
    Write-Host $line -ForegroundColor Cyan
    Write-Host ""
}

function Write-Step($text) {
    Write-Host "  >> $text" -ForegroundColor Yellow
}

function Write-Ok($text) {
    Write-Host "  [OK] $text" -ForegroundColor Green
}

function Write-Fail($text) {
    Write-Host "  [FAIL] $text" -ForegroundColor Red
}

function Get-Timestamp {
    return (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
}

function Get-Ratio($gen) {
    # Relative gen index: 1 = BaseRatio, 2 = BaseRatio + RatioStep, etc.
    if ($gen -lt 1) { $gen = 1 }  # Guard against gen=0
    $ratio = $BaseRatio + ($gen - 1) * $RatioStep
    # Cap at 0.50 to keep general data majority
    return [Math]::Min($ratio, 0.50)
}


# ---- Periodic ETA ----

$script:pipelinePhase   = ""           # Current phase label  (e.g. "Self-play Gen 3")
$script:phaseStart      = $null        # When current phase began
$script:genTimings      = @()          # Per-gen timing records for ETA projection
                                        # Each entry: @{ Secs=<float>; EpochsRan=<int>; EpochsMax=<int>; Normalized=<float> }

function Format-Duration([TimeSpan]$ts) {
    if ($ts.TotalHours -ge 1) {
        return "{0}h {1:00}m {2:00}s" -f [int][Math]::Floor($ts.TotalHours), $ts.Minutes, $ts.Seconds
    } elseif ($ts.TotalMinutes -ge 1) {
        return "{0}m {1:00}s" -f [int][Math]::Floor($ts.TotalMinutes), $ts.Seconds
    } elseif ($ts.TotalSeconds -lt 1) {
        return "{0:N1}s" -f $ts.TotalSeconds
    } else {
        return "{0}s" -f $ts.Seconds
    }
}

function Write-ETA {
    <#  Prints a pipeline ETA banner in bright magenta.
        Shows:  current gen / total gens | phase elapsed
                pipeline remaining estimate | projected finish clock time  #>
    param([string]$Phase, [datetime]$PhaseStart, [datetime]$PipelineStart,
          [int]$GensCompleted, [int]$GensTotal)

    $now          = Get-Date
    $phaseElapsed = $now - $PhaseStart
    $totalElapsed = $now - $PipelineStart

    $genLabel     = "Gen $($GensCompleted + 1)/$GensTotal"
    $peStr        = Format-Duration $phaseElapsed
    $teStr        = Format-Duration $totalElapsed

    # Line 1: Current activity
    $line1 = "[ETA] $Phase ($genLabel) | Phase elapsed: $peStr | Total elapsed: $teStr"

    # Line 2: Pipeline projection using normalized gen timings (early-stop aware)
    $line2 = ""
    if ($script:genTimings.Count -gt 0 -and $GensTotal -gt $GensCompleted) {
        # Use normalized times: early-stopped gens are scaled up to full-epoch estimate
        $actualAvg     = ($script:genTimings | ForEach-Object { $_.Normalized  # INFO [12.12]: Use normalized time (accounts for early-stop) for accurate ETA } | Measure-Object -Average).Average
        $gensLeft      = $GensTotal - $GensCompleted
        $remSec        = ($actualAvg * $gensLeft) - $phaseElapsed.TotalSeconds
        if ($remSec -lt 0) { $remSec = 0 }
        $remStr        = Format-Duration ([TimeSpan]::FromSeconds($remSec))
        $finishTime    = $now.AddSeconds($remSec)
        # Show if any early stops were detected
        $esCount       = ($script:genTimings | Where-Object { $_.EpochsRan -lt $_.EpochsMax }).Count
        $esNote        = if ($esCount -gt 0) { " ($esCount early-stop adj.)" } else { "" }
        $line2         = "     Pipeline remaining: ~$remStr | Finish: ~$($finishTime.ToString('HH:mm'))$esNote"
    } else {
        $line2         = "     (pipeline projection available after gen 1 completes)"
    }

    $bar = "=" * 70
    Write-Host ""
    Write-Host "  $bar" -ForegroundColor Magenta
    Write-Host "  $line1"              -ForegroundColor Magenta
    Write-Host "  $line2"              -ForegroundColor Magenta
    Write-Host "  $bar" -ForegroundColor Magenta
    Write-Host ""
}

function Run-WithETA {
    <#  Runs an external process and prints ETA every $ETAIntervalMin minutes.
        The child inherits the parent console so output streams directly to
        the terminal and Ctrl+C is delivered correctly (no orphan processes).  #>
    param(
        [string]$Exe,
        [string[]]$Arguments,
        [string]$Phase,
        [datetime]$GenStart = (Get-Date),
        [datetime]$PipelineStart = (Get-Date),
        [int]$GensCompleted = 0
    )

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName               = $Exe
    # FIX 12.4 + 4.22: Complete argument escaping — handle trailing backslashes,
    # embedded quotes, and PowerShell metacharacters ($, `, ')
    $psi.Arguments              = ($Arguments | ForEach-Object {
        if ($_ -match '["\s&|><(){}^%!;,=''$`]') {
            $escaped = $_ -replace '(\\*)$', '$1$1'      # Double trailing backslashes
            $escaped = $escaped -replace '(\\*)"', '$1$1\"'  # Escape backslashes before quotes
            "`"$escaped`""
        } else { $_ }
    }) -join " "
    $psi.UseShellExecute        = $false
    # No redirection — child writes directly to the console and inherits
    # the console window so Ctrl+C / CTRL_C_EVENT reaches it.
    $psi.RedirectStandardOutput = $false
    $psi.RedirectStandardError  = $false
    $psi.CreateNoWindow         = $false

    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo = $psi
    $proc.Start() | Out-Null

    $intervalMs   = $ETAIntervalMin * 60 * 1000
    $phaseStart   = if ($GenStart) { $GenStart } else { Get-Date }
    $lastETA      = Get-Date

    try {
        while (-not $proc.HasExited) {
            $null = $proc.WaitForExit(2000)   # poll every 2 s

            $elapsed = ((Get-Date) - $lastETA).TotalMilliseconds
            if ($elapsed -ge $intervalMs) {
                Write-ETA -Phase $Phase -PhaseStart $phaseStart `
                          -PipelineStart $PipelineStart `
                          -GensCompleted $GensCompleted -GensTotal $Generations
                $lastETA = Get-Date
            }
        }
    } finally {
        # On Ctrl+C, ensure child process finishes or is killed (prevents zombies)
        if (-not $proc.HasExited) {
            Write-Host "`n[Pipeline] Waiting for child process to finish..." -ForegroundColor Yellow
            if (-not $proc.WaitForExit(5000)) {
                Write-Host "[Pipeline] Force-killing child process" -ForegroundColor Red
                $proc.Kill()
                $proc.WaitForExit(3000)
            }
        }
    }

    # Ensure the process is fully finished
    $proc.WaitForExit()

    return $proc.ExitCode
}

# ============================================================================
#  Overfitting Detection
# ============================================================================

function Test-Overfitting {
    <#  Reads training_log.csv, filters to the most recent run_id, and prints
        coloured health warnings if any overfitting signals are detected.

        Signals checked (in order of severity):
          CRIT  - NaN or Inf in any loss value
          CRIT  - Loss explosion: any epoch where loss > 2x the previous epoch loss
          HIGH  - val_loss > train_loss * 1.10 for 5+ consecutive epochs
          WARN  - val_loss > train_loss * 1.05 for 3+ consecutive epochs
          INFO  - Train loss falling consistently while val loss is flat or rising
                  (early sign of diverging generalisation gap)
          OK    - No signals found
    #>
    param([string]$LogPath)

    if (-not (Test-Path $LogPath)) {
        Write-Host "  [Health] No training log found, skipping overfitting check." -ForegroundColor DarkGray
        return
    }

    # Read CSV
    try {
        $rows = Import-Csv $LogPath
    } catch {
        Write-Host "  [Health] Could not parse training log: $_" -ForegroundColor DarkGray
        return
    }

    if ($rows.Count -lt 2) {
        Write-Host "  [Health] Not enough epochs to analyse (need at least 2)." -ForegroundColor DarkGray
        return
    }

    # Filter to the current run_id (last entry)
    $currentRunId = $rows[-1].run_id
    $run = @($rows | Where-Object { $_.run_id -eq $currentRunId })

    if ($run.Count -lt 2) {
        Write-Host "  [Health] Only 1 epoch in current run, skipping." -ForegroundColor DarkGray
        return
    }

    # Parse numeric columns; skip rows where parsing fails
    $parsed = @()
    foreach ($r in $run) {
        try {
            $tl = [double]$r.loss
            $vl = [double]$r.val_loss
            $parsed += @{ Train = $tl; Val = $vl }
        } catch { <# silently skip malformed rows #> }
    }

    if ($parsed.Count -lt 2) { return }

    $bar = "-" * 60
    $anyIssue = $false

    Write-Host ""
    Write-Host "  $bar" -ForegroundColor DarkCyan
    Write-Host "  [Health Check]  $($parsed.Count) epochs analysed (run $currentRunId)" -ForegroundColor DarkCyan
    Write-Host "  $bar" -ForegroundColor DarkCyan

    # --- CRIT: NaN / Inf ---
    $nanRows = @($parsed | Where-Object { [Double]::IsNaN($_.Train) -or [Double]::IsInfinity($_.Train) -or
                                           [Double]::IsNaN($_.Val)   -or [Double]::IsInfinity($_.Val) })
    if ($nanRows.Count -gt 0) {
        Write-Host "  [CRIT] NaN or Inf detected in loss values! Training may have collapsed." -ForegroundColor Red
        $anyIssue = $true
    }

    # --- CRIT: Loss explosion (>2x spike between consecutive epochs) ---
    $exploded = $false
    for ($k = 1; $k -lt $parsed.Count; $k++) {
        $prev = $parsed[$k - 1].Train
        $curr = $parsed[$k].Train
        if ($prev -gt 0 -and $curr -gt ($prev * 2.0)) {
            Write-Host ("  [CRIT] Loss explosion at epoch {0}: {1:F6} -> {2:F6} ({3:F1}x spike)" -f `
                ($k + 1), $prev, $curr, ($curr / $prev)) -ForegroundColor Red
            $exploded = $true
            $anyIssue = $true
        }
    }

    # --- Count consecutive epochs where val > train * threshold ---
    function Count-ConsecutiveTail {
        param([object[]]$Data, [double]$Threshold)
        $count = 0
        for ($k = $Data.Count - 1; $k -ge 0; $k--) {
            $t = $Data[$k].Train
            $v = $Data[$k].Val
            if ($t -gt 0 -and $v -gt 0 -and $v -gt ($t * $Threshold)) {
                $count++
            } else {
                break
            }
        }
        return $count
    }

    $highCount = Count-ConsecutiveTail -Data $parsed -Threshold 1.10
    $warnCount = Count-ConsecutiveTail -Data $parsed -Threshold 1.05

    if ($highCount -ge 5) {
        $lastTrain = $parsed[-1].Train
        $lastVal   = $parsed[-1].Val
        $ratio     = if ($lastTrain -gt 0) { $lastVal / $lastTrain } else { 0 }
        Write-Host ("  [HIGH] Overfitting: val_loss > train_loss*1.10 for {0} consecutive epochs " +
                    "(ratio {1:F3}x). Consider reducing LR or increasing WeightDecay." -f $highCount, $ratio) -ForegroundColor Red
        $anyIssue = $true
    } elseif ($warnCount -ge 3) {
        $lastTrain = $parsed[-1].Train
        $lastVal   = $parsed[-1].Val
        $ratio     = if ($lastTrain -gt 0) { $lastVal / $lastTrain } else { 0 }
        Write-Host ("  [WARN] Overfitting: val_loss > train_loss*1.05 for {0} consecutive epochs " +
                    "(ratio {1:F3}x). Monitor closely next generation." -f $warnCount, $ratio) -ForegroundColor Yellow
        $anyIssue = $true
    }

    # --- INFO: Diverging generalisation gap (train falling, val flat/rising) ---
    # Measure over the last half of epochs (at least 4 epochs)
    $window = [Math]::Max(4, [int][Math]::Floor($parsed.Count / 2))
    if ($parsed.Count -ge $window) {
        $tail       = $parsed[($parsed.Count - $window)..($parsed.Count - 1)]
        $trainFirst = $tail[0].Train
        $trainLast  = $tail[-1].Train
        $valFirst   = $tail[0].Val
        $valLast    = $tail[-1].Val

        $trainDrop  = if ($trainFirst -gt 0) { ($trainFirst - $trainLast) / $trainFirst } else { 0 }
        $valChange  = if ($valFirst   -gt 0) { ($valLast - $valFirst) / $valFirst }       else { 0 }

        # Train dropped >3% but val rose or barely moved (<0.5% drop)
        if ($trainDrop -gt 0.03 -and $valChange -gt -0.005) {
            Write-Host ("  [INFO] Generalisation gap widening: train fell {0:P1} but val changed {1:+.1%;-.1%;0%} " +
                        "over last {2} epochs. Early overfitting signal." -f $trainDrop, $valChange, $window) -ForegroundColor Yellow
            $anyIssue = $true
        }
    }

    # --- Summary stats ---
    $finalTrain = $parsed[-1].Train
    $finalVal   = $parsed[-1].Val
    $bestVal    = ($parsed | ForEach-Object { $_.Val } | Measure-Object -Minimum).Minimum
    $ratio      = if ($finalTrain -gt 0 -and $finalVal -gt 0) { $finalVal / $finalTrain } else { 0 }

    Write-Host ("  Train loss (final): {0:F6}  |  Val loss (final): {1:F6}  |  Best val: {2:F6}  |  Ratio: {3:F3}x" -f `
        $finalTrain, $finalVal, $bestVal, $ratio) -ForegroundColor DarkCyan

    if (-not $anyIssue) {
        Write-Host "  [OK] No overfitting signals detected. Training looks healthy." -ForegroundColor Green
    } else {
        # Actionable recommendations based on what fired
        Write-Host ""
        Write-Host "  Suggested remedies:" -ForegroundColor DarkYellow
        if ($warnCount -ge 3 -or $highCount -ge 5) {
            Write-Host "    - Increase -WeightDecay (current: $WeightDecay, try 0.001 or 0.005)" -ForegroundColor DarkYellow
            Write-Host "    - Increase -Dropout (current: $Dropout, try 0.15 or 0.2)" -ForegroundColor DarkYellow
            Write-Host "    - Reduce -LR (current: $LR, try halving it)" -ForegroundColor DarkYellow
            Write-Host "    - Add -LabelSmoothing 0.05 to soften result targets" -ForegroundColor DarkYellow
        }
        if ($exploded) {
            Write-Host "    - Reduce -LR significantly (explosion usually means LR too high)" -ForegroundColor DarkYellow
            Write-Host "    - Check -GradClip is enabled (current: $GradClip)" -ForegroundColor DarkYellow
        }
    }

    Write-Host "  $bar" -ForegroundColor DarkCyan
    Write-Host ""
}

# ============================================================================
#  Validation
# ============================================================================

# --- Parameter validation ---
if ($Generations -le 0) { Write-Error "Generations must be > 0"; exit 1 }
if ($GamesPerGen -le 0) { Write-Error "GamesPerGen must be > 0"; exit 1 }
if ($Depth -le 0) { Write-Error "Depth must be > 0"; exit 1 }
if ($Workers -le 0) { Write-Error "Workers must be > 0"; exit 1 }
if ($MaxPositions -le 0) { Write-Error "MaxPositions must be > 0"; exit 1 }
if ($Epochs -le 0) { Write-Error "Epochs must be > 0"; exit 1 }
if ($BatchSize -le 0) { Write-Error "BatchSize must be > 0"; exit 1 }
if ($LR -le 0) { Write-Error "LR must be > 0"; exit 1 }
if ($LRMin -le 0) { Write-Error "LRMin must be > 0"; exit 1 }
if ($GradClip -le 0) { Write-Error "GradClip must be > 0"; exit 1 }
if ($Dropout -lt 0 -or $Dropout -ge 1) { Write-Error "Dropout must be in [0, 1)"; exit 1 }
if ($LabelSmoothing -lt 0 -or $LabelSmoothing -ge 1) { Write-Error "LabelSmoothing must be in [0, 1)"; exit 1 }
if ($Openings -ne "" -and -not (Test-Path $Openings)) {
    Write-Error "Openings file not found: $Openings"
    exit 1
}

Write-Banner "NNUE Training Pipeline"
$firstGen = $StartGen + 1
$lastGen  = $StartGen + $Generations
Write-Host "  Current gen      : $StartGen"
Write-Host "  Generations      : $Generations  (Gen $firstGen -> Gen $lastGen)"
Write-Host "  Games per gen    : $GamesPerGen"
Write-Host "  Depth            : $Depth"
Write-Host "  Workers          : $Workers"
Write-Host "  Max positions    : $MaxPositions"
Write-Host "  Epochs           : $Epochs"
Write-Host "  Early stop       : $EarlyStop"
Write-Host "  Batch size       : $BatchSize"
Write-Host ""
Write-Host "  -- Hyperparameters --" -ForegroundColor Cyan
Write-Host "  Learning rate    : $LR" -ForegroundColor Cyan
Write-Host "  LR minimum       : $LRMin  (cosine decay floor)" -ForegroundColor Cyan
Write-Host "  Weight decay     : $WeightDecay  (L2 regularisation)" -ForegroundColor Cyan
Write-Host "  Grad clip        : $GradClip  (exploding gradient guard)" -ForegroundColor Cyan
Write-Host "  Dropout          : $Dropout" -ForegroundColor Cyan
Write-Host "  Label smoothing  : $LabelSmoothing" -ForegroundColor Cyan
Write-Host "  Draw weight      : $DrawWeight"
Write-Host "  Mate boost       : $MateBoost"
Write-Host "  Base ratio       : $BaseRatio"
Write-Host "  Ratio step       : $RatioStep"
if ($Openings -ne "") {
    Write-Host "  Openings file    : $Openings"
}
Write-Host "  Validate ELO     : $ValidateElo"
Write-Host "  ETA interval     : every ${ETAIntervalMin}min" -ForegroundColor Magenta

# Auto-calculate warmup steps if not set (--enhanced uses grad_accum=4)
# NOTE: Warmup steps calculated from MaxPositions (requested), not actual loaded count.
# If dataset has fewer positions, warmup extends beyond intended 3 epochs.
# TODO: Have train_nnue.py report actual dataset size for accurate warmup.
if ($WarmupSteps -lt 0) {
    $stepsPerEpoch = [Math]::Ceiling($MaxPositions / ($BatchSize * $GradAccum))
    $WarmupSteps = 3 * $stepsPerEpoch
    Write-Host "  Warmup steps     : $WarmupSteps (auto: ~3 epochs)" -ForegroundColor Magenta
} else {
    Write-Host "  Warmup steps     : $WarmupSteps" -ForegroundColor Magenta
}
Write-Host ""

# Check engine exists
if (-not (Test-Path $Engine)) {
    Write-Fail "Engine not found at: $Engine"
    Write-Host "  Build in Release mode first, or pass -Engine <path>"
    exit 1
}

# Check training data exists
if (-not (Test-Path $TrainingData)) {
    Write-Fail "Training data not found at: $TrainingData"
    exit 1
}

# Check starting gen weights exist (first loop iteration needs gen$StartGen weights)
$startWeights = "$AssetsDir\nnue_weights_gen$StartGen.bin"
if (-not (Test-Path $startWeights)) {
    # Fall back to nnue_weights.bin
    if (Test-Path "$AssetsDir\nnue_weights.bin") {
        Write-Step "Gen$StartGen weights not found, backing up current nnue_weights.bin as gen$StartGen"
        Copy-Item "$AssetsDir\nnue_weights.bin" $startWeights
        Write-Ok "Created $startWeights"
    } else {
        Write-Fail "No starting weights found. Need $startWeights or $AssetsDir\nnue_weights.bin"
        exit 1
    }
}

# Check Python
$pyCheck = & py -$PythonVersion --version 2>&1  # F6.3: use configurable version
if ($LASTEXITCODE -ne 0) {
    Write-Fail "Python $PythonVersion not found."
    exit 1
}

# ============================================================================
#  Log file
# ============================================================================

$logFile = "$AssetsDir\pipeline_log.txt"
function Log($msg) {
    $entry = "$(Get-Timestamp) | $msg"
    Add-Content -Path $logFile -Value $entry
    Write-Host "  $entry" -ForegroundColor DarkGray
}

Log "Pipeline started: Gen $firstGen -> Gen $lastGen"

# ============================================================================
#  Clear training log once at start (unless PlotPerGen, which clears each gen)
# ============================================================================

$trainingLog = "training_log.csv"
if (-not $PlotPerGen) {
    if (Test-Path $trainingLog) {
        Remove-Item $trainingLog
        Write-Step "Cleared training log for fresh cumulative plot"
    }
}

# ============================================================================
#  Main Loop
# ============================================================================

$totalStart = Get-Date

for ($i = 0; $i -lt $Generations; $i++) {
    $gen = $StartGen + 1 + $i
    $prevGen = $gen - 1
    $ratio = Get-Ratio ($i + 1)   # Use relative gen index (1-based) so ratio starts from BaseRatio regardless of StartGen
    $prevWeights = "$AssetsDir\nnue_weights_gen$prevGen.bin"
    $selfplayFile = "$AssetsDir\selfplay_gen$gen.bin"
    $outputWeights = "$AssetsDir\nnue_weights.bin"
    $genWeights = "$AssetsDir\nnue_weights_gen$gen.bin"

    $genStart = Get-Date

    Write-Banner "Generation $gen / $lastGen  (self-play ratio: $ratio)"

    # Print pipeline ETA at generation start
    if ($i -gt 0) {
        Write-ETA -Phase "Starting Gen $gen" -PhaseStart $genStart `
                  -PipelineStart $totalStart `
                  -GensCompleted $i -GensTotal $Generations
    }

    # ---- Step 1: Self-Play ----
    Write-Step "Self-play: $GamesPerGen games at depth $Depth with $Workers workers"
    Log "Gen $gen - Self-play start ($GamesPerGen games, depth $Depth)"

    $selfplayArgs = @("--generate", "--games", $GamesPerGen, "--depth", $Depth, "--workers", $Workers, "--output", $selfplayFile)

    if ($Openings -ne "" -and (Test-Path $Openings)) {
        $selfplayArgs += @("--openings", $Openings)
        Write-Step "Using openings file: $Openings"
    }

    $spExit = Run-WithETA -Exe $Engine -Arguments $selfplayArgs -Phase "Self-play Gen $gen" -GenStart $genStart -PipelineStart $totalStart -GensCompleted $i

    if ($spExit -ne 0 -or -not (Test-Path $selfplayFile)) {
        Write-Fail "Self-play failed for Gen $gen"
        Log "Gen $gen - Self-play FAILED (exit code: $spExit)"
        exit 1
    }

    # FIX 4.4: Validate self-play output file is not truncated/corrupted
    $selfplaySizeBytes = (Get-Item $selfplayFile).Length
    $selfplaySize = $selfplaySizeBytes / 1KB
    $minExpectedBytes = [long]$GamesPerGen * 30 * 20  # conservative: ~30 positions/game, ~20 bytes/position minimum
    if ($selfplaySizeBytes -lt $minExpectedBytes) {
        Write-Fail "Self-play output file too small: $([Math]::Round($selfplaySize, 0)) KB (expected at least $([Math]::Round($minExpectedBytes / 1KB, 0)) KB) — possible corruption or crash mid-write"
        Log "Gen $gen - Self-play VALIDATION FAILED: file too small ($selfplaySizeBytes bytes < $minExpectedBytes bytes expected)"
        exit 1
    }
    Write-Ok "Self-play complete: $selfplayFile ($([Math]::Round($selfplaySize, 0)) KB, validated)"
    Log "Gen $gen - Self-play complete ($([Math]::Round($selfplaySize, 0)) KB)"

    # ---- Step 2: Training ----
    Write-Step "Training: loading gen$prevGen weights, ratio $ratio, max $MaxPositions positions"
    Log "Gen $gen - Training start (ratio: $ratio, max-pos: $MaxPositions, epochs: $Epochs)"

    # Build training args
    # NOTE: --extra-data uses nargs='*' in argparse. Argument order matters —
    # inserting new args between --extra-data and its values would break parsing.
    $trainArgs = @(
        "train_nnue.py",
        "--data", $TrainingData,
        "--extra-data", $selfplayFile, $ratio,
        "--max-positions", $MaxPositions,
        "--epochs", $Epochs,
        "--batch-size", $BatchSize,
        "--enhanced",
        "--swa",
        "--early-stop", $EarlyStop,
        "--weight-decay", $WeightDecay,
        "--lr", $LR,
        "--lr-min", $LRMin,
        "--grad-clip", $GradClip,
        "--dropout", $Dropout,
        "--load-weights", $prevWeights,
        "--output", $outputWeights,
        "--no-cosine-restarts",
        "--draw-weight", $DrawWeight,
        "--mate-boost", $MateBoost,
        "--warmup-steps", $WarmupSteps,
        "--grad-accum", $GradAccum,
        "--plot"
    )

    # Label smoothing: only pass if non-zero (keeps default behaviour when unset)
    if ($LabelSmoothing -gt 0) {
        $trainArgs += "--label-smoothing"
        $trainArgs += $LabelSmoothing
    }

    # Only open the plot image on the final generation
    $isLastGen = ($i -eq ($Generations - 1))
    if ($isLastGen) {
        $trainArgs += "--show-plot"
    }

    # PlotPerGen: clear log before each generation for individual plots
    if ($PlotPerGen) {
        $trainArgs += "--clear-log"
    }

    $pyExe  = (Get-Command py).Source
    $pyArgs = @("-$PythonVersion", "-u") + $trainArgs  # F6.3: use configurable version
    $trExit = Run-WithETA -Exe $pyExe -Arguments $pyArgs -Phase "Training Gen $gen" -GenStart $genStart -PipelineStart $totalStart -GensCompleted $i

    if ($trExit -ne 0) {
        Write-Fail "Training failed for Gen $gen"
        Log "Gen $gen - Training FAILED (exit code: $trExit)"
        exit 1
    }

    Write-Ok "Training complete"
    Log "Gen $gen - Training complete"

    # ---- Step 2b: Overfitting Health Check ----
    Test-Overfitting -LogPath $trainingLog

    # PlotPerGen: save generation-specific plot into "training progress" folder
    $plotDir = Join-Path "." "training progress"
    if (-not (Test-Path $plotDir)) { New-Item -ItemType Directory -Path $plotDir | Out-Null }
    $plotFile = Join-Path $plotDir "training_progress.png"
    if ($PlotPerGen -and (Test-Path $plotFile)) {
        $genPlot = Join-Path $plotDir "training_progress_gen$gen.png"
        Copy-Item $plotFile $genPlot
        Write-Ok "Saved plot: $genPlot"
    }

    # ---- Step 3: Save generation weights ----
    if (-not (Test-Path $outputWeights)) {
        Write-Fail "Training produced no output weights at $outputWeights"
        exit 1
    }
    $tmpWeights = "$genWeights.tmp"
    Copy-Item $outputWeights $tmpWeights -ErrorAction Stop
    Move-Item $tmpWeights $genWeights -Force
    Write-Ok "Saved weights: $genWeights"
    Log "Gen $gen - Weights saved to $genWeights"

    # ---- Step 4: Optional ELO Validation ----
    if ($ValidateElo) {
        $cutechess = "cutechess\cutechess-cli.exe"
        if (Test-Path $cutechess) {
            Write-Step "Validation: $ValidationGames games Gen$gen vs Gen$prevGen"
            Log "Gen $gen - Validation start ($ValidationGames games)"

            $newEngine = "cmd=uci_engine.bat"
            $prevEngine = "cmd=uci_engine.bat"

            # Run quick match
            $valArgs = @(
                "-engine", "name=Gen$gen", $newEngine, "option.WeightsFile=$genWeights",
                "-engine", "name=Gen$prevGen", $prevEngine, "option.WeightsFile=$prevWeights",
                "-each", "proto=uci", "tc=1+0.1",
                "-rounds", $ValidationGames,
                "-pgnout", "$AssetsDir\validation_gen${gen}.pgn",
                "-recover"
            )
            $valResult = Run-WithETA -Exe $cutechess -Arguments $valArgs -Phase "Validation Gen $gen" -GenStart $genStart -PipelineStart $totalStart -GensCompleted $i
            if ($valResult -ne 0) {
                Write-Fail "Validation match failed with exit code $valResult"
                exit 1
            }

            Write-Ok "Validation complete (see output above)"
            Log "Gen $gen - Validation complete"
        } else {
            Write-Step "Skipping validation (cutechess-cli not found at $cutechess)"
        }
    }

    # ---- Generation Summary ----
    $genElapsed = (Get-Date) - $genStart
    $genSecs    = $genElapsed.TotalSeconds
    $genTime    = Format-Duration $genElapsed

    # Detect actual epochs by counting data rows for THIS gen's run_id
    $epochsRan = $Epochs   # assume full run by default
    if (Test-Path $trainingLog) {
        $logData = Import-Csv $trainingLog
        if ($logData.Count -gt 0) {
            $lastEntry = $logData[-1]
            $currentRunId = $lastEntry.run_id
            $epochsRan = @($logData | Where-Object { $_.run_id -eq $currentRunId }).Count
        }
    }
    $earlyStopHit = $epochsRan -lt $Epochs

    # Normalize: scale up early-stopped gens to estimate full-epoch time
    # FIX 4.16: NOTE — This assumes linear per-epoch timing (each epoch takes equal time).
    # In practice, later epochs may be slower (SWA overhead, validation, etc.), so the
    # projected time may underestimate actual full-training duration.
    if ($earlyStopHit -and $epochsRan -gt 0) {
        $normalized = $genSecs * ($Epochs / $epochsRan)
    } else {
        $normalized = $genSecs
    }
    $script:genTimings += @{
        Secs      = $genSecs
        EpochsRan = $epochsRan
        EpochsMax = $Epochs
        Normalized = $normalized
    }

    # Display summary
    $esLabel = if ($earlyStopHit) { " (early stop at epoch $epochsRan/$Epochs)" } else { " (full $Epochs epochs)" }
    Write-Banner "Gen $gen complete in $genTime$esLabel"
    Log "Gen $gen - Completed in $genTime - epochs $epochsRan/$Epochs"

    # ETA for remaining generations using normalized averages
    $gensRemaining = $Generations - ($i + 1)
    if ($gensRemaining -gt 0) {
        $actualAvg     = ($script:genTimings | ForEach-Object { $_.Normalized  # INFO [12.12]: Use normalized time (accounts for early-stop) for accurate ETA } | Measure-Object -Average).Average
        $etaSeconds    = $actualAvg * $gensRemaining
        $eta           = Format-Duration ([TimeSpan]::FromSeconds($etaSeconds))
        $finishTime    = (Get-Date).AddSeconds($etaSeconds)
        $esCount       = ($script:genTimings | Where-Object { $_.EpochsRan -lt $_.EpochsMax }).Count
        $esNote        = if ($esCount -gt 0) { " | $esCount gen(s) normalized for early stop" } else { "" }
        Write-Host "  ETA remaining    : ~$eta ($gensRemaining gen(s) left) | Finish: ~$($finishTime.ToString('HH:mm'))$esNote" -ForegroundColor Magenta
    }
}

# ============================================================================
#  Final Summary
# ============================================================================

$totalElapsed = (Get-Date) - $totalStart
$totalTime = Format-Duration $totalElapsed

Write-Banner "Pipeline Complete!"
Write-Host "  Generations trained : $firstGen -> $lastGen"
Write-Host "  Total time          : $totalTime"
Write-Host "  Latest weights      : $AssetsDir\nnue_weights_gen$lastGen.bin"
Write-Host "  Log file            : $logFile"
Write-Host ""
Log "Pipeline complete: $Generations generations in $totalTime"
