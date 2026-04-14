# ============================================================
# find_elo.ps1  -  Calibrate engine Elo vs Stockfish
# ============================================================
# Method: native CLI Elo adjustment
#   engineElo = testElo + (Elo diff reported by cutechess-cli)
#   Round 1  : coarse estimate at starting Elo
#   Round 2  : one refinement pass if round-1 gap > $RefineThreshold
#   Phase 2  : verification rounds at the converged estimate
#
# ChessEngine.exe is a Windows-subsystem (GUI) app.  The --uci flag
# tells it to enter UCI protocol mode and connect stdin/stdout to
# the pipes created by cutechess-cli.  No console window is opened.
# ============================================================

# ---- FIX #13: CmdletBinding + parameterised paths (was hardcoded) ----
# ---- FIX #6 : Paths derived from $PSScriptRoot with overridable params ----
[CmdletBinding()]
param(
    [string]$ReleaseDir   = $(if ($PSScriptRoot -match 'x64[/\\]Release$') { $PSScriptRoot } else { Join-Path $PSScriptRoot "x64\Release" }),
    [string]$ProjectRoot  = $PSScriptRoot,

    [int]   $CalGames        = 50,       # Games per calibration round
    [int]   $VerifyGames     = 100,      # Games per verification round
    [int]   $VerifyRounds    = 2,        # Verification rounds after calibration
    [double]$CalMoveTime     = 1,        # Seconds per move during calibration
    [double]$VerifyMoveTime  = 2,        # Seconds per move during verification (more accurate)
    [int]   $ProbeMoveTimeMs = 1000,     # Milliseconds per move for engine probes (UCI sanity checks)
    [int]   $Concurrency     = 12,        # Parallel games (keep <= CPU cores)
    [int]   $DefaultElo      = 1500,     # Starting guess
    [int]   $RefineThreshold = 50,       # Re-run if first adjustment was larger than this
    [int]   $VerifyEloBand   = 30,       # Verification passes if |Elo diff| is within this

    # FIX #19: Optional opening book file for cutechess-cli
    [string]$OpeningBook     = "",       # Path to opening book (PGN or EPD) for -openings

    # Manual SF Elo override -- skip calibration, jump straight to verification
    [int]   $SfEloOverride   = 0,        # Set >0 to skip calibration and verify at this Elo

    # FIX #2: Keep PGN files after run for post-mortem inspection
    [switch]$KeepPGN
)

# ---- FIX #14: Strict error handling ----
$ErrorActionPreference = 'Stop'

# ---- FIX #16: Transcript logging ----
$LogFile = Join-Path $ReleaseDir "elo_calibration_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
try { Start-Transcript -Path $LogFile -Append | Out-Null }
catch { Write-Warning "Could not start transcript logging to $LogFile -- continuing without log." }

# ---- CONFIG (derived from params) ------------------------------------
$MyEngineCmd  = Join-Path $ReleaseDir  "ChessEngine.exe"
$StockfishCmd = Join-Path $ProjectRoot "stockfish\stockfish.exe"
$CutechessCmd = Join-Path $ProjectRoot "cutechess\cutechess-cli.exe"

$EloFile          = Join-Path $ReleaseDir "last_confirmed_elo.txt"
# FIX #17: Separate file for unverified estimates
$EstimateEloFile  = Join-Path $ReleaseDir "last_estimate_elo.txt"

$AssetsDir = Join-Path $ReleaseDir "assets"

# FIX #1: Move SF Elo range constants to CONFIG block (was defined after Run-Match)
$SF_ELO_MIN = 1320
$SF_ELO_MAX = 3190

# FIX #4: Cap synthetic Elo diff to prevent wild swings from floor/ceiling fallback
$MaxSyntheticEloDiff = 400
# ----------------------------------------------------------------------

# ---- FIX #8: Validate concurrency vs CPU cores ----
$cpuCores = 0
try {
    $cpuCores = (Get-CimInstance Win32_Processor -ErrorAction SilentlyContinue |
                 Measure-Object -Property NumberOfLogicalProcessors -Sum).Sum
} catch { }
if ($cpuCores -gt 0 -and $Concurrency -gt $cpuCores) {
    Write-Warning "Concurrency ($Concurrency) exceeds logical CPU cores ($cpuCores). This may cause time forfeits due to I/O contention."
    Write-Warning "Consider setting -Concurrency to $cpuCores or lower."
}

# ---- FIX #15: Renamed Kill-ProcessTree -> Stop-ProcessTree (approved verb) ----
function Stop-ProcessTree([int]$ParentPid) {
    # NOTE (Audit #9): This queries all system processes via WMI. Acceptable for
    # cleanup of a single process tree, but would be slow if called frequently.
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.ParentProcessId -eq $ParentPid } |
        ForEach-Object { Stop-ProcessTree $_.ProcessId }
    Stop-Process -Id $ParentPid -Force -ErrorAction SilentlyContinue
}

$script:cutechessProc = $null

# NOTE (Audit #10): PowerShell.Exiting only fires on graceful shutdown.
# Hard kills (Task Manager, Stop-Process) will bypass this handler.
# The trap block below handles Ctrl+C, which is the common interactive case.
$null = Register-EngineEvent PowerShell.Exiting -Action {
    if ($script:cutechessProc -and -not $script:cutechessProc.HasExited) {
        Stop-ProcessTree $script:cutechessProc.Id
    }
}

trap {
    if ($script:cutechessProc -and -not $script:cutechessProc.HasExited) {
        Write-Host "`n  Interrupted -- cleaning up engine processes..." -ForegroundColor Yellow
        Stop-ProcessTree $script:cutechessProc.Id
    }
    Write-Host "  Exiting." -ForegroundColor Yellow
    try { Stop-Transcript | Out-Null } catch {}
    # FIX #18: Exit with distinct code for interruption
    exit 3
}

# ============================================================
# DIAGNOSTIC: UCI self-test
# Spawns the engine with --uci, sends the UCI handshake, and
# verifies it responds correctly before wasting time on matches.
# ============================================================
function Test-EngineUci {
    param([string]$EngineCmd, [string]$WeightsFile)

    Write-Host ""
    Write-Host "  [Pre-flight] Testing engine UCI handshake..." -ForegroundColor DarkCyan -NoNewline

    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName  = $EngineCmd
    # FIX #7: Renamed $args_list -> $uciArgs to avoid proximity to automatic $args
    $uciArgs = @("--uci")
    if ($WeightsFile) { $uciArgs += "--weights"; $uciArgs += $WeightsFile }
    $psi.Arguments = ($uciArgs | ForEach-Object {
        if ($_ -match '\s') { "`"$_`"" } else { $_ }
    }) -join ' '
    $psi.UseShellExecute        = $false
    $psi.RedirectStandardInput  = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError  = $true
    $psi.CreateNoWindow         = $true

    try {
        $proc = [System.Diagnostics.Process]::Start($psi)
    } catch {
        Write-Host " FAILED" -ForegroundColor Red
        Write-Host "  ERROR: Could not start engine process: $_" -ForegroundColor Red
        Write-Host "  >> Check that '$EngineCmd' exists and is a valid executable." -ForegroundColor Yellow
        return $false
    }

    # Drain stderr asynchronously to prevent pipe-buffer deadlock
    $stderrTask = $proc.StandardError.ReadToEndAsync()

    # Send "uci" and read lines with a watchdog timer (no Peek -- avoids pipe blocking)
    $proc.StandardInput.WriteLine("uci")
    $proc.StandardInput.Flush()

    $gotUciOk   = $false
    $gotReadyOk = $false
    $outputLines = @()

    # Phase 1: wait up to 5s for uciok
    $watchdog = [System.Diagnostics.Stopwatch]::StartNew()
    $uciTimeout = 5000
    while ($watchdog.ElapsedMilliseconds -lt $uciTimeout -and -not $gotUciOk) {
        $readTask = $proc.StandardOutput.ReadLineAsync()
        while (-not $readTask.IsCompleted) {
            if ($watchdog.ElapsedMilliseconds -ge $uciTimeout -or $proc.HasExited) { break }
            Start-Sleep -Milliseconds 20
        }
        if ($readTask.IsCompleted -and $readTask.Result -ne $null) {
            $line = $readTask.Result
            $outputLines += $line
            if ($line.Trim() -eq "uciok") { $gotUciOk = $true }
        } else { break }
    }

    # Phase 2: send isready, wait up to 30s for readyok
    if ($gotUciOk) {
        $proc.StandardInput.WriteLine("isready")
        $proc.StandardInput.Flush()

        $watchdog.Restart()
        $readyTimeout = 30000
        while ($watchdog.ElapsedMilliseconds -lt $readyTimeout -and -not $gotReadyOk) {
            $readTask = $proc.StandardOutput.ReadLineAsync()
            while (-not $readTask.IsCompleted) {
                if ($watchdog.ElapsedMilliseconds -ge $readyTimeout -or $proc.HasExited) { break }
                Start-Sleep -Milliseconds 20
            }
            if ($readTask.IsCompleted -and $readTask.Result -ne $null) {
                $line = $readTask.Result
                $outputLines += $line
                if ($line.Trim() -eq "readyok") { $gotReadyOk = $true }
            } else { break }
        }
    }

    # Clean up
    try { $proc.StandardInput.WriteLine("quit") } catch {}
    try { $proc.WaitForExit(2000) | Out-Null } catch {}
    if (-not $proc.HasExited) { Stop-ProcessTree $proc.Id }

    $stderrContent = ""
    try { $stderrContent = $stderrTask.Result } catch {}

    if ($gotUciOk -and $gotReadyOk) {
        Write-Host " OK (uciok + readyok received)" -ForegroundColor Green
        return $true
    }

    # Failed -- show exactly what we got
    Write-Host " FAILED" -ForegroundColor Red
    Write-Host ""
    Write-Host "  !! Engine UCI self-test failed. This is likely why all games are lost." -ForegroundColor Red
    Write-Host ""

    if (-not $gotUciOk -and $outputLines.Count -eq 0 -and $proc.HasExited) {
        Write-Host "  >> Engine exited immediately with no output." -ForegroundColor Yellow
        Write-Host "     Possible causes:" -ForegroundColor Yellow
        Write-Host "       1. Missing weights file -- engine crashes on load" -ForegroundColor Yellow
        if ($WeightsFile) {
            $exists = Test-Path $WeightsFile
            Write-Host "          Weights path: $WeightsFile" -ForegroundColor $(if ($exists) { "Gray" } else { "Red" })
            if (-not $exists) {
                Write-Host "          *** FILE NOT FOUND ***" -ForegroundColor Red
            }
        }
        Write-Host "       2. Missing DLL or Visual C++ redistributable" -ForegroundColor Yellow
        Write-Host "       3. --uci flag not implemented or crashes the engine" -ForegroundColor Yellow
        Write-Host "     Try running manually: & '$EngineCmd' --uci" -ForegroundColor Cyan
    } elseif (-not $gotUciOk) {
        Write-Host "  >> Engine started but did not send 'uciok'." -ForegroundColor Yellow
        Write-Host "     Output received ($($outputLines.Count) lines):" -ForegroundColor Yellow
        $outputLines | Select-Object -First 10 | ForEach-Object { Write-Host "       $_" -ForegroundColor Gray }
        Write-Host "     Possible causes:" -ForegroundColor Yellow
        Write-Host "       1. Engine opens a GUI window instead of entering UCI mode" -ForegroundColor Yellow
        Write-Host "       2. --uci argument is not being parsed before GUI init" -ForegroundColor Yellow
        Write-Host "       3. Engine sends 'id name/author' but forgets 'uciok'" -ForegroundColor Yellow
    } elseif (-not $gotReadyOk) {
        Write-Host "  >> Engine sent 'uciok' but timed out waiting for 'readyok'." -ForegroundColor Yellow
        Write-Host "     Possible causes:" -ForegroundColor Yellow
        Write-Host "       1. Engine is taking too long to load the weights file" -ForegroundColor Yellow
        Write-Host "       2. 'isready' command not implemented or causes a hang" -ForegroundColor Yellow
    }

    if ($stderrContent.Trim()) {
        Write-Host ""
        Write-Host "  Engine stderr output:" -ForegroundColor Yellow
        $stderrContent -split "`n" | Select-Object -First 15 | ForEach-Object {
            $t = $_.TrimEnd()
            if ($t) { Write-Host "    $t" -ForegroundColor DarkYellow }
        }
    }

    Write-Host ""
    Write-Host "  Continuing anyway -- match results will diagnose further." -ForegroundColor DarkGray
    return $false
}

# ---- FIX #15: Renamed Find-BestGenWeights -> Get-BestGenWeights (approved verb) ----
function Get-BestGenWeights {
    param([string]$AssetsDir)

    # Collect available gen numbers from .bin files (needed for LatestGen + fallback)
    $genFiles = Get-ChildItem -Path $AssetsDir -Filter "nnue_weights_gen*.bin" -ErrorAction SilentlyContinue
    $genNums  = @()
    foreach ($f in $genFiles) {
        if ($f.Name -match "^nnue_weights_gen(\d+)\.bin$") {
            $genNums += [int]$Matches[1]
        }
    }
    $genNums = $genNums | Sort-Object
    if ($genNums.Count -eq 0) { return $null }

    $latestGen = $genNums[-1]

    # ---- Primary: read gen_stats.csv (same source as C++ loadBestGenFromFile) ----
    # Format: gen,val_loss  one row per gen, one entry = best val_loss seen for that gen
    $statsPath = Join-Path $AssetsDir "gen_stats.csv"
    if (Test-Path $statsPath) {
        $invCulture = [System.Globalization.CultureInfo]::InvariantCulture
        $floatStyle = [System.Globalization.NumberStyles]::Float
        $bestGen     = -1
        $bestValLoss = [double]::MaxValue

        foreach ($line in [System.IO.File]::ReadAllLines($statsPath)) {
            $line = $line.Trim()
            if ($line -eq '') { continue }
            $comma = $line.IndexOf(',')
            if ($comma -lt 1) { continue }
            $gRaw = $line.Substring(0, $comma).Trim()
            $vRaw = $line.Substring($comma + 1).Trim()
            $g = 0; $vl = 0.0
            if ([int]::TryParse($gRaw, [ref]$g) -and
                [double]::TryParse($vRaw, $floatStyle, $invCulture, [ref]$vl) -and
                $vl -gt 0) {
                if ($vl -lt $bestValLoss) {
                    $bestValLoss = $vl
                    $bestGen     = $g
                }
            }
        }

        if ($bestGen -ge 0) {
            return @{ Gen = $bestGen; ValLoss = $bestValLoss; LatestGen = $latestGen }
        }
        # gen_stats.csv existed but had no valid rows -- fall through to fallback
    }

    # ---- Fallback: no gen_stats.csv -- return null so caller uses latest gen ----
    return $null
}

# ---- Resolve which weights file to use ----
$ResolvedWeightsFile = $null

$bestResult = Get-BestGenWeights -AssetsDir $AssetsDir
if ($bestResult) {
    $bestGenNum   = $bestResult.Gen
    $latestGenNum = $bestResult.LatestGen
    $ResolvedWeightsFile = Join-Path $AssetsDir "nnue_weights_gen$bestGenNum.bin"
    if ($bestGenNum -eq $latestGenNum) {
        Write-Host "  Weights          : Gen $bestGenNum (best val_loss = latest gen)" -ForegroundColor Green
    } else {
        Write-Host "  Weights          : Gen $bestGenNum [best val_loss $([Math]::Round($bestResult.ValLoss,6))]  (latest: gen $latestGenNum)" -ForegroundColor Green
    }
} else {
    $latestFile = Get-ChildItem -Path $AssetsDir -Filter "nnue_weights_gen*.bin" -ErrorAction SilentlyContinue |
                  Where-Object { $_.Name -match "^nnue_weights_gen(\d+)\.bin$" } |
                  Sort-Object { [int]($_.BaseName -replace '[^\d]','') } |
                  Select-Object -Last 1
    if ($latestFile) {
        $latestGenNum = [int]($latestFile.BaseName -replace '[^\d]','')
        $ResolvedWeightsFile = $latestFile.FullName
        Write-Host "  Weights          : Gen $latestGenNum (latest gen, training_log unavailable)" -ForegroundColor Yellow
    } else {
        Write-Host "  Weights          : engine default (no gen files found)" -ForegroundColor Yellow
    }
}

# Validate tools
# FIX #18: Distinct exit codes for different failure modes
if (-not (Test-Path $MyEngineCmd))  { Write-Error "Engine not found: $MyEngineCmd";          exit 1 }
if (-not (Test-Path $StockfishCmd)) { Write-Error "Stockfish not found: $StockfishCmd";      exit 1 }
if (-not (Test-Path $CutechessCmd)) { Write-Error "cutechess-cli not found: $CutechessCmd";  exit 1 }

# FIX #19: Validate opening book if specified
if ($OpeningBook -and -not (Test-Path $OpeningBook)) {
    Write-Warning "Opening book not found: $OpeningBook -- games will start from default position."
    $OpeningBook = ""
}

# Warn if weights file was resolved but the file is missing on disk
if ($ResolvedWeightsFile -and -not (Test-Path $ResolvedWeightsFile)) {
    Write-Host ""
    Write-Host "  WARNING: Resolved weights file does not exist on disk:" -ForegroundColor Red
    Write-Host "    $ResolvedWeightsFile" -ForegroundColor Red
    Write-Host "  >> The engine will likely crash on startup and lose every game." -ForegroundColor Yellow
    Write-Host "  >> Check that training completed successfully and the file was saved." -ForegroundColor Yellow
}

# ---- Probe engine: start a position, capture raw UCI traffic, check bestmove ----
# FIX #15: Renamed Probe-EngineMove -> Test-EngineMove (approved verb)
function Test-EngineMove {
    param(
        [string]$EngineCmd,
        [string]$WeightsFile,
        [string]$PositionCmd,   # e.g. "position startpos moves a2a4"
        [string]$Label,         # human-readable description of the position
        [int]$MoveTimeMs = 500
    )

    Write-Host ""
    Write-Host "  [Probe] $Label" -ForegroundColor DarkCyan
    Write-Host "          -> $PositionCmd" -ForegroundColor DarkGray

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName               = $EngineCmd
    $psi.UseShellExecute        = $false
    $psi.RedirectStandardInput  = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError  = $true
    $psi.CreateNoWindow         = $true
    # FIX 12.13: Escape double quotes in weights file path to prevent argument injection
    $safeWeights = if ($WeightsFile) { $WeightsFile -replace '"', '\"' } else { $null }
    $psi.Arguments = if ($safeWeights) { "--uci --weights `"$safeWeights`"" } else { "--uci" }

    $proc = [System.Diagnostics.Process]::Start($psi)
    if (-not $proc) {
        Write-Host "  [Probe] Failed to start engine." -ForegroundColor Red
        return
    }

    $allLines = [System.Collections.Generic.List[string]]::new()

    # Watchdog timer: kills engine if it doesn't finish within total budget.
    # NOTE (Audit #5): Timer.Elapsed fires on a ThreadPool thread. The closure
    # captures $proc. In practice, killing the process causes ReadLine() to
    # return $null (EOF), which breaks the main loop. A theoretical race exists
    # but is benign -- the worst case is a redundant Kill() call that throws
    # (caught by the try/catch).
    $wdMs = $MoveTimeMs + 15000   # handshake (10s) + search + margin
    $wd   = New-Object System.Timers.Timer($wdMs)
    $wd.AutoReset = $false
    $wd.add_Elapsed({ try { if (-not $proc.HasExited) { $proc.Kill() } } catch {} }.GetNewClosure())
    $wd.Start()

    # Read-Until: blocking readline loop; returns first line matching $Pattern, or $null on EOF/kill
    function Read-Until { param([string]$Pattern)
        while ($true) {
            $line = $proc.StandardOutput.ReadLine()
            if ($null -eq $line) { return $null }   # EOF or process killed
            $allLines.Add($line)
            if ($line.Trim() -match $Pattern) { return $line }
        }
    }

    # Handshake
    $proc.StandardInput.WriteLine("uci");     $proc.StandardInput.Flush()
    $null = Read-Until '^uciok$'

    $proc.StandardInput.WriteLine("isready"); $proc.StandardInput.Flush()
    $readyLine = Read-Until '^readyok$'

    if (-not $readyLine) {
        Write-Host "  [Probe] Engine not ready (timed out)." -ForegroundColor Red
        $wd.Stop(); $wd.Dispose()
        try { $proc.WaitForExit(500) | Out-Null } catch {}
        if (-not $proc.HasExited) { Stop-ProcessTree $proc.Id }
        return
    }

    # Reset watchdog to search-only budget, then send position + go
    $wd.Stop()
    $wd.Interval = $MoveTimeMs + 5000
    $wd.Start()

    $proc.StandardInput.WriteLine($PositionCmd)
    $proc.StandardInput.WriteLine("go movetime $MoveTimeMs")
    $proc.StandardInput.Flush()

    # Blocking read for bestmove
    $bestLine = Read-Until '^bestmove\s+\S+'
    $bestmove = $null
    if ($bestLine -and $bestLine -match "^bestmove\s+(\S+)") { $bestmove = $Matches[1] }

    $wd.Stop(); $wd.Dispose()
    try { $proc.StandardInput.WriteLine("quit") } catch {}
    try { $proc.WaitForExit(1000) | Out-Null } catch {}
    if (-not $proc.HasExited) { Stop-ProcessTree $proc.Id }

    # Show last info line and bestmove
    $lastInfo = $allLines | Where-Object { $_ -match "^info.*score" } | Select-Object -Last 1
    if ($lastInfo) { Write-Host "          Last info: $lastInfo" -ForegroundColor DarkGray }

    if ($bestmove) {
        Write-Host "          bestmove : $bestmove" -ForegroundColor Cyan
        if ($bestmove -match "^d[1-8]d[1-8]$") {
            Write-Host "          !! bestmove moves a d-file piece -- matches suspected bug pattern" -ForegroundColor Red
        }
    } else {
        Write-Host "          bestmove : (none received -- engine timed out or crashed)" -ForegroundColor Red
        Write-Host "          Raw output:" -ForegroundColor Yellow
        $allLines | Select-Object -Last 8 | ForEach-Object { Write-Host "            $_" -ForegroundColor DarkGray }
    }
}

# ---- Parse PGN: count game termination reasons ----
# FIX #15: Renamed Analyze-PGN -> Get-PGNAnalysis (approved verb)
function Get-PGNAnalysis {
    param([string]$PgnPath)

    if (-not (Test-Path $PgnPath)) { return }

    $content = Get-Content -Path $PgnPath -Raw -ErrorAction SilentlyContinue
    if (-not $content) { return }

    # Split into individual games (each starts with [Event)
    $games = ($content -split '(?=\[Event )') | Where-Object { $_.Trim() }

    $counts = @{
        IllegalMove = 0
        TimeForfeit = 0
        Adjudicated = 0
        Normal      = 0
        Other       = 0
    }
    $illegalMoveDetails = [System.Collections.Generic.List[string]]::new()
    $gameMoveCounts     = [System.Collections.Generic.List[int]]::new()

    foreach ($game in $games) {
        # Extract Termination tag
        $term = ""
        if ($game -match '\[Termination "([^"]+)"\]') { $term = $Matches[1] }

        # FIX #11: Count half-moves played (approximate -- regex-based, may miscount
        # edge cases like O-O or e.p. notation; treat as rough indicator only)
        $moveSection = $game -replace '\[[^\]]+\]', ''  # strip headers
        $moveSection = $moveSection -replace '\{[^}]*\}', ''  # strip comments
        $moveTokens  = ($moveSection -split '\s+') | Where-Object { $_ -match '^\d+\.' -or $_ -match '^[a-zA-Z][a-zA-Z0-9\-+#=x]+$' -or $_ -match '^[a-h][1-8][a-h][1-8]' }
        $halfMoves   = ($moveTokens | Where-Object { $_ -notmatch '^\d+\.' }).Count

        # Find illegal move details from comments
        $illegalDetail = ""
        if ($game -match 'illegal move[^}]*\(([^)]+)\)') {
            $illegalDetail = $Matches[1]
        } elseif ($game -match 'illegal move[^}]*\b([a-h][1-8][a-h][1-8][qrbn]?)\b') {
            $illegalDetail = $Matches[1]
        }

        if ($term -match "illegal|Illegal") {
            $counts.IllegalMove++
            $detail = if ($illegalDetail) { " (move: $illegalDetail, at approx half-move $halfMoves)" } else { " (at approx half-move $halfMoves)" }
            $illegalMoveDetails.Add($detail)
        } elseif ($term -match "time|Time") {
            $counts.TimeForfeit++
        } elseif ($term -match "adjudication|Adjudication") {
            $counts.Adjudicated++
        } elseif ($term -match "normal|Normal") {
            $counts.Normal++
        } else {
            $counts.Other++
        }
    }

    $total = $games.Count
    Write-Host ""
    Write-Host "     PGN analysis ($total games in file):" -ForegroundColor DarkCyan
    if ($counts.IllegalMove -gt 0) {
        Write-Host ("       Illegal move losses : {0}" -f $counts.IllegalMove) -ForegroundColor Red
        # Show first few distinct illegal move details
        $shown = $illegalMoveDetails | Select-Object -Unique | Select-Object -First 6
        foreach ($d in $shown) {
            Write-Host "         $d" -ForegroundColor DarkYellow
        }
        # Cluster: how early in the game are illegal moves happening?
        $earlyCount = ($illegalMoveDetails | Where-Object { $_ -match "half-move ([0-9]+)" -and [int]$Matches[1] -lt 20 }).Count
        if ($earlyCount -gt 0) {
            Write-Host ("         -> {0}/{1} happen before move 10 (opening phase)" -f $earlyCount, $counts.IllegalMove) -ForegroundColor Yellow
            Write-Host "            Likely a static/initialization bug, not a search bug." -ForegroundColor Yellow
        }
    }
    if ($counts.TimeForfeit  -gt 0) { Write-Host ("       Time forfeits       : {0}" -f $counts.TimeForfeit)  -ForegroundColor Yellow }
    if ($counts.Adjudicated  -gt 0) { Write-Host ("       Adjudicated         : {0}" -f $counts.Adjudicated)  -ForegroundColor DarkGray }
    if ($counts.Normal       -gt 0) { Write-Host ("       Normal endings      : {0}" -f $counts.Normal)       -ForegroundColor DarkGray }
    if ($counts.Other        -gt 0) { Write-Host ("       Other/unknown       : {0}" -f $counts.Other)        -ForegroundColor DarkGray }
}

# Run UCI self-test now that paths are validated
$uciOk = Test-EngineUci -EngineCmd $MyEngineCmd -WeightsFile $ResolvedWeightsFile

if ($uciOk) {
    # FIX #3: Corrected label -- startpos is White to play, not Black
    Test-EngineMove -EngineCmd $MyEngineCmd -WeightsFile $ResolvedWeightsFile `
        -PositionCmd "position startpos" `
        -Label "Starting position (White to play: e2e4/d2d4 would be typical)" `
        -MoveTimeMs $ProbeMoveTimeMs

    # Probe 2: after 1.e4 -- Black to play; a sane reply is d7d5, e7e5, etc.
    Test-EngineMove -EngineCmd $MyEngineCmd -WeightsFile $ResolvedWeightsFile `
        -PositionCmd "position startpos moves e2e4" `
        -Label "After 1.e4 (Black to play: e7e5/d7d5/c7c5 would be typical)" `
        -MoveTimeMs $ProbeMoveTimeMs

    # Probe 3: replicate a PV-like sequence -- 3 half-moves so Black to play
    Test-EngineMove -EngineCmd $MyEngineCmd -WeightsFile $ResolvedWeightsFile `
        -PositionCmd "position startpos moves a2a4 b8c6 b2b5" `
        -Label "After 1.a4 Nc6 2.b5 (Black to play -- does engine pick a sane reply?)" `
        -MoveTimeMs $ProbeMoveTimeMs
    Write-Host ""
}

# FIX #2: exit code for UCI failure
if (-not $uciOk) {
    Write-Warning "UCI self-test failed. Proceeding with matches, but results may be unreliable."
}

# ---- Run a match and return structured result ----
function Invoke-Match {
    param([int]$TestElo, [int]$NumGames, [string]$Label, [double]$MoveTime)

    $halfRounds = [Math]::Max(1, [int]($NumGames / 2))

    # Stockfish only accepts UCI_Elo 1320-3190. If our estimate is below that,
    # clamp to 1320 and note it -- the score/Elo-diff calculation still works
    # because we measure against a known SF strength and apply the diff afterward.
    $sfElo = [Math]::Max($SF_ELO_MIN, [Math]::Min($SF_ELO_MAX, $TestElo))
    if ($sfElo -ne $TestElo) {
        Write-Host ("  [Note] Requested SF Elo {0} is outside Stockfish range {1}-{2} -- clamped to {3}." -f $TestElo, $SF_ELO_MIN, $SF_ELO_MAX, $sfElo) -ForegroundColor DarkYellow
        Write-Host ("         Elo diff from this match will be applied relative to SF {0}." -f $sfElo) -ForegroundColor DarkYellow
        # FIX #12: Warn about reliability at extreme Elo values
        if ($TestElo -lt ($SF_ELO_MIN - 200)) {
            Write-Host "         WARNING: Engine estimate is far below SF minimum. Calibration at this range is inherently unreliable." -ForegroundColor Yellow
        }
    }

    $myEngineTokens = @(
        "name=MyEngine",
        "cmd=$MyEngineCmd",
        "arg=--uci"
    )
    if ($ResolvedWeightsFile) {
        $myEngineTokens += "arg=--weights"
        $myEngineTokens += "arg=$ResolvedWeightsFile"
    }
    $myEngineTokens += "proto=uci"

    $cutechessArgs = @("-engine") + $myEngineTokens + @(
        "-engine", "name=Stockfish", "cmd=$StockfishCmd", "proto=uci",
                   "option.UCI_LimitStrength=true",
                   "option.UCI_Elo=$sfElo",
        "-each",   "st=$MoveTime", "timemargin=2000",
        "-rounds", "$halfRounds", "-games", "2", "-repeat",
        "-concurrency", "$Concurrency",
        "-recover",
        "-pgnout", (Join-Path $ReleaseDir "elo_cal_${Label}.pgn")
    )

    # FIX #19: Add opening book if specified
    if ($OpeningBook) {
        $bookExt = [System.IO.Path]::GetExtension($OpeningBook).ToLower()
        $bookFormat = if ($bookExt -eq ".epd") { "epd" } else { "pgn" }
        $cutechessArgs += @("-openings", "file=$OpeningBook", "format=$bookFormat", "order=random")
        Write-Host "  [Note] Using opening book: $OpeningBook (format: $bookFormat)" -ForegroundColor DarkGray
    }

    $outputLines      = [System.Collections.Generic.List[string]]::new()
    $gamesFinished    = 0
    $liveW = 0; $liveL = 0; $liveD = 0

    # Diagnostic counters
    $timeForfeitCount  = 0   # "loses on time" / "time forfeit"
    $crashCount        = 0   # "disconnected" / "connection stalled" / "forfeits"
    $illegalMoveCount  = 0   # "illegal move"

    $sw = [System.Diagnostics.Stopwatch]::StartNew()

    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $CutechessCmd
    $psi.Arguments = ($cutechessArgs | ForEach-Object {
        if ($_ -match '\s') { "`"$_`"" } else { $_ }
    }) -join ' '
    $psi.UseShellExecute        = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError  = $true
    $psi.CreateNoWindow         = $true

    Write-Host "  Launching cutechess-cli ($NumGames games, ${MoveTime}s/move, concurrency $Concurrency)..." -ForegroundColor DarkGray
    Write-Verbose "  CMD: $($psi.FileName) $($psi.Arguments)"

    $proc = [System.Diagnostics.Process]::Start($psi)
    $script:cutechessProc = $proc

    $stderrTask = $proc.StandardError.ReadToEndAsync()

    $firstResultShown = $false
    try {
        while ($null -ne ($line = $proc.StandardOutput.ReadLine())) {
            $outputLines.Add($line)

            if (-not $firstResultShown -and $line -match "^Finished game") {
                $firstResultShown = $true
                Write-Host "  First game completed after $([Math]::Round($sw.Elapsed.TotalSeconds, 1))s -- results streaming in." -ForegroundColor DarkGray
            }

            # Running score -- the Score line is the authoritative progress source.
            # With concurrency > 1, "Finished game N" can arrive out of order
            # (e.g. game 8 before game 7), so we derive gamesFinished from W+L+D
            # which always equals the true number of completed games.
            if ($line -match "^Score of MyEngine vs Stockfish:\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)") {
                $liveW = [int]$Matches[1]
                $liveL = [int]$Matches[2]
                $liveD = [int]$Matches[3]
                $gamesFinished = $liveW + $liveL + $liveD

                $elapsed = $sw.Elapsed
                $pct = [Math]::Round(($gamesFinished / $NumGames) * 100)
                $etaStr = ""
                if ($gamesFinished -gt 0 -and $gamesFinished -lt $NumGames) {
                    $secsPerGame = $elapsed.TotalSeconds / $gamesFinished
                    $remaining   = [int]($secsPerGame * ($NumGames - $gamesFinished))
                    $etaMins     = [int][Math]::Floor($remaining / 60)
                    $etaSecs     = $remaining % 60
                    $etaStr      = "  ETA: ${etaMins}m ${etaSecs}s"
                }
                $scoreStr = "  ${liveW}W/${liveL}L/${liveD}D"
                Write-Host "`r  Progress: $gamesFinished / $NumGames games ($pct%)$scoreStr$etaStr    " -NoNewline -ForegroundColor DarkCyan
            }

            # ---- Diagnostic pattern matching ----
            # Time forfeit (engine ran out of time on a move)
            if ($line -match "MyEngine.*loses on time|MyEngine.*time forfeit|forfeit.*MyEngine.*time") {
                $timeForfeitCount++
            }
            # Engine crash / disconnect (engine process died or pipe broke)
            if ($line -match "MyEngine.*disconnected|MyEngine.*connection stalled|MyEngine.*forfeits.*connection|MyEngine.*loses.*adjudication") {
                $crashCount++
            }
            # Illegal move (engine sent a move that isn't legal)
            if ($line -match "MyEngine.*illegal move|illegal move.*MyEngine") {
                $illegalMoveCount++
            }
        }

        $proc.WaitForExit()
    }
    finally {
        if (-not $proc.HasExited) {
            Write-Host "`n  Cleaning up child processes..." -ForegroundColor Yellow
            Stop-ProcessTree $proc.Id
        }
        $script:cutechessProc = $null
    }

    $stderrOutput = $stderrTask.Result
    $sw.Stop()

    # Final progress line
    if ($gamesFinished -gt 0) {
        $totalSecs    = [int]$sw.Elapsed.TotalSeconds
        $totalMins    = [int][Math]::Floor($totalSecs / 60)
        $totalRemSecs = $totalSecs % 60
        $finalScore   = "  ${liveW}W/${liveL}L/${liveD}D"
        Write-Host "`r  Progress: $NumGames / $NumGames games (100%)$finalScore  Time: ${totalMins}m ${totalRemSecs}s    " -ForegroundColor DarkCyan
    }

    # Append stderr lines
    if ($stderrOutput) {
        $stderrOutput -split "`n" | ForEach-Object {
            $trimmed = $_.TrimEnd()
            if ($trimmed) { $outputLines.Add($trimmed) }
        }
    }

    $output = $outputLines.ToArray()

    # --- Parse score line ---
    $wins = 0; $losses = 0; $draws = 0; $score = 0.5
    $scoreLine = $output | Where-Object { $_ -match "^Score of MyEngine vs Stockfish:" } | Select-Object -Last 1
    if ($scoreLine -match "(\d+)\s*-\s*(\d+)\s*-\s*(\d+)\s*\[([0-9.]+)\]") {
        $wins   = [int]$Matches[1]
        $losses = [int]$Matches[2]
        $draws  = [int]$Matches[3]
        $score  = [double]$Matches[4]
    } else {
        Write-Host ""
        Write-Host "  ERROR: Could not parse score line from cutechess output." -ForegroundColor Red
        Write-Host "  >> Last 10 lines of cutechess output:" -ForegroundColor Yellow
        $output | Select-Object -Last 10 | ForEach-Object { Write-Host "     $_" -ForegroundColor Gray }
        if ($stderrOutput.Trim()) {
            Write-Host "  >> cutechess stderr:" -ForegroundColor Yellow
            $stderrOutput -split "`n" | Select-Object -First 10 | ForEach-Object {
                $t = $_.TrimEnd(); if ($t) { Write-Host "     $t" -ForegroundColor DarkYellow }
            }
        }
        Write-Host "  >> Ensure cutechess-cli can communicate with ChessEngine.exe --uci" -ForegroundColor Yellow
        return $null
    }

    # --- Diagnostic summary (shown when result looks suspicious) ---
    $totalGames  = $wins + $losses + $draws
    $allLost     = ($wins -eq 0 -and $draws -eq 0 -and $losses -gt 0)
    $lopsided    = ($totalGames -gt 0 -and $losses -gt 0 -and ($losses / $totalGames) -ge 0.9)

    if ($allLost -or $lopsided -or $timeForfeitCount -gt 0 -or $crashCount -gt 0 -or $illegalMoveCount -gt 0) {
        Write-Host ""
        Write-Host "  !! Suspicious result -- diagnostic summary:" -ForegroundColor Red

        if ($timeForfeitCount -gt 0) {
            Write-Host "     Time forfeits   : $timeForfeitCount game(s)" -ForegroundColor Yellow
            Write-Host "       >> Engine is thinking too long for ${MoveTime}s/move." -ForegroundColor Yellow
            Write-Host "          Try reducing `$CalMoveTime (currently ${MoveTime}s) or `$Concurrency (currently $Concurrency)." -ForegroundColor Yellow
        }

        if ($crashCount -gt 0) {
            Write-Host "     Engine crashes  : $crashCount game(s) (disconnected/stalled)" -ForegroundColor Yellow
            Write-Host "       >> Engine process is dying during play." -ForegroundColor Yellow
            Write-Host "          Check for access violations, missing DLLs, or assert failures." -ForegroundColor Yellow
        }

        if ($illegalMoveCount -gt 0) {
            Write-Host "     Illegal moves   : $illegalMoveCount game(s)" -ForegroundColor Yellow
            Write-Host "       >> Engine is sending moves that are not legal in the position." -ForegroundColor Yellow
            Write-Host "          Likely a move-generation or UCI parsing bug." -ForegroundColor Yellow
            # Parse the saved PGN for per-game detail
            $pgnPath = Join-Path $ReleaseDir "elo_cal_${Label}.pgn"
            Get-PGNAnalysis -PgnPath $pgnPath
        }

        if ($timeForfeitCount -eq 0 -and $crashCount -eq 0 -and $illegalMoveCount -eq 0 -and $allLost) {
            Write-Host "     No forfeits or crashes detected in cutechess output." -ForegroundColor DarkGray
            if (-not $uciOk) {
                Write-Host "       >> UCI self-test failed earlier -- engine may not be responding" -ForegroundColor Yellow
                Write-Host "          to cutechess commands, causing adjudication losses." -ForegroundColor Yellow
            } else {
                Write-Host "       >> All losses appear to be genuine chess losses." -ForegroundColor DarkGray
                Write-Host "          Engine may simply be weaker than Elo 1000 at this training stage." -ForegroundColor DarkGray
                Write-Host "          Consider training more generations before Elo testing." -ForegroundColor DarkGray
            }
        }

        # Detect UCI_Elo rejection specifically -- means SF played at full strength
        if ($stderrOutput -match "Invalid value for option UCI_Elo") {
            Write-Host ""
            Write-Host "  !! Stockfish rejected the UCI_Elo setting:" -ForegroundColor Red
            Write-Host "       Stockfish only accepts UCI_Elo 1320-3190." -ForegroundColor Yellow
            Write-Host "       If you see this, the script has a clamping bug -- please report it." -ForegroundColor Yellow
        }

        # Show relevant stderr lines from cutechess (engine init errors often appear here)
        $relevantStderr = $stderrOutput -split "`n" |
            Where-Object { $_ -match "error|Error|ERROR|fatal|assert|crash|warn|Warn" } |
            ForEach-Object { $_.TrimEnd() } |
            Where-Object { $_ }
        if ($relevantStderr) {
            Write-Host ""
            Write-Host "     cutechess stderr (errors/warnings):" -ForegroundColor Yellow
            $relevantStderr | Select-Object -First 15 | ForEach-Object {
                Write-Host "       $_" -ForegroundColor DarkYellow
            }
        }
        Write-Host ""
    }

    # --- Parse Elo diff from cutechess ---
    $eloDiff  = $null
    $eloError = $null
    $eloLine  = $output | Where-Object { $_ -match "Elo difference:" } | Select-Object -Last 1

    if ($eloLine -match "Elo difference:\s*([-0-9.]+)\s*\+/-\s*(nan|[0-9.]+)") {
        $eloDiff  = [double]$Matches[1]
        $eloError = if ($Matches[2] -ne "nan") { [double]$Matches[2] } else { $null }
    }

    # Fallback when cutechess returns nan or no Elo line
    if ($null -eq $eloDiff -or ($eloDiff -eq 0.0 -and $score -ne 0.5)) {
        if ($score -le 0.0) {
            $floorScore = 0.5 / [Math]::Max(1, $NumGames)
            $eloDiff = [Math]::Round(-400.0 * [Math]::Log10(1.0 / $floorScore - 1.0), 1)
            # FIX #4: Cap synthetic Elo diff to prevent wild swings
            $eloDiff = [Math]::Max(-$MaxSyntheticEloDiff, $eloDiff)
            Write-Host "  (Score=0: all losses -- synthetic Elo diff capped to $eloDiff)" -ForegroundColor DarkGray
        } elseif ($score -ge 1.0) {
            $ceilScore = 1.0 - 0.5 / [Math]::Max(1, $NumGames)
            $eloDiff = [Math]::Round(-400.0 * [Math]::Log10(1.0 / $ceilScore - 1.0), 1)
            # FIX #4: Cap synthetic Elo diff to prevent wild swings
            $eloDiff = [Math]::Min($MaxSyntheticEloDiff, $eloDiff)
            Write-Host "  (Score=1: all wins -- synthetic Elo diff capped to $eloDiff)" -ForegroundColor DarkGray
        } else {
            $eloDiff = [Math]::Round(-400.0 * [Math]::Log10(1.0 / $score - 1.0), 1)
            Write-Host "  (Elo diff estimated from score=$score, cutechess returned nan)" -ForegroundColor DarkGray
        }
    }

    return @{
        Wins           = $wins
        Losses         = $losses
        Draws          = $draws
        Score          = $score
        Total          = $wins + $losses + $draws
        EloDiff        = $eloDiff
        EloError       = $eloError
        TimeForfeit    = $timeForfeitCount
        Crashes        = $crashCount
        IllegalMoves   = $illegalMoveCount
    }
}

# ---- FIX #15: Renamed Clamp-Elo -> Limit-Elo (approved verb) ----
function Limit-Elo([int]$elo) {
    return [Math]::Max($SF_ELO_MIN, [Math]::Min($SF_ELO_MAX, $elo))
}

# ---- Load last confirmed Elo (or use default) ----
$startElo = $DefaultElo
if (Test-Path $EloFile) {
    $saved = Get-Content $EloFile -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($saved -match "^-?\d+$") {  # FIX 12.5: accept negative Elo values
        $startElo = [int]$saved
        Write-Host "  Loaded last confirmed Elo: $startElo" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  Engine Elo Calibration  (native CLI Elo-adjustment method)"    -ForegroundColor Cyan
Write-Host "  Starting point : $startElo"                                     -ForegroundColor Cyan
Write-Host "  Calibration    : up to 2 rounds x $CalGames games @ ${CalMoveTime}s/move"    -ForegroundColor Cyan
Write-Host "  Verification   : $VerifyRounds rounds x $VerifyGames games @ ${VerifyMoveTime}s/move" -ForegroundColor Cyan
Write-Host "  Concurrency    : $Concurrency"                                  -ForegroundColor Cyan
if ($SfEloOverride -gt 0) {
    Write-Host "  SF Elo override: $SfEloOverride (skipping calibration)"      -ForegroundColor Yellow
}
if ($OpeningBook) {
    Write-Host "  Opening book   : $OpeningBook"                               -ForegroundColor Cyan
}
Write-Host "  Log file       : $LogFile"                                       -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan

# ============================================================
# PHASE 1 -- Calibration (skipped when -SfEloOverride is set)
# ============================================================
if ($SfEloOverride -gt 0) {
    $candidateElo = Limit-Elo $SfEloOverride
    Write-Host ""
    Write-Host "=== CALIBRATION SKIPPED -- manual override ===" -ForegroundColor Yellow
    Write-Host "  Using SF Elo override: $SfEloOverride (clamped to $candidateElo)" -ForegroundColor Yellow
    Write-Host "  Jumping straight to verification." -ForegroundColor Yellow
} else {
    $candidateElo = Limit-Elo $startElo

    Write-Host ""
    Write-Host "=== CALIBRATION Round 1 | SF Elo: $candidateElo ===" -ForegroundColor Yellow

    $r1 = Invoke-Match -TestElo $candidateElo -NumGames $CalGames -Label "cal_r1" -MoveTime $CalMoveTime
    if (-not $r1) { try { Stop-Transcript | Out-Null } catch {}; exit 1 }

    $adjustment1  = [int][Math]::Round($r1.EloDiff)
    $candidateElo = Limit-Elo ($candidateElo + $adjustment1)

    $errStr = if ($r1.EloError) { "+/- $($r1.EloError)" } else { "+/- n/a" }
    Write-Host ""
    Write-Host "  Result    : $($r1.Wins)W / $($r1.Losses)L / $($r1.Draws)D | Score: $($r1.Score)" -ForegroundColor Green
    Write-Host "  CLI diff  : $($r1.EloDiff) $errStr  =>  new estimate: $candidateElo" -ForegroundColor Green

    if ([Math]::Abs($adjustment1) -gt $RefineThreshold) {
        Write-Host ""
        Write-Host "=== CALIBRATION Round 2 (refinement) | SF Elo: $candidateElo ===" -ForegroundColor Yellow

        $r2 = Invoke-Match -TestElo $candidateElo -NumGames $CalGames -Label "cal_r2" -MoveTime $CalMoveTime
        if (-not $r2) { try { Stop-Transcript | Out-Null } catch {}; exit 1 }

        $adjustment2  = [int][Math]::Round($r2.EloDiff)
        $candidateElo = Limit-Elo ($candidateElo + $adjustment2)

        $errStr2 = if ($r2.EloError) { "+/- $($r2.EloError)" } else { "+/- n/a" }
        Write-Host ""
        Write-Host "  Result    : $($r2.Wins)W / $($r2.Losses)L / $($r2.Draws)D | Score: $($r2.Score)" -ForegroundColor Green
        Write-Host "  CLI diff  : $($r2.EloDiff) $errStr2  =>  refined estimate: $candidateElo" -ForegroundColor Green
    } else {
        Write-Host "  Adjustment small ($adjustment1) -- skipping refinement round." -ForegroundColor DarkGray
    }
}

# ============================================================
# PHASE 2 -- Verification
# ============================================================
Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  VERIFICATION  |  Testing at Elo $candidateElo"                 -ForegroundColor Cyan
Write-Host "  $VerifyRounds rounds x $VerifyGames games @ ${VerifyMoveTime}s/move" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan

$allEloDiffs = @()
$verified    = $true

for ($v = 1; $v -le $VerifyRounds; $v++) {
    Write-Host ""
    Write-Host "=== VERIFICATION Round $v ===" -ForegroundColor Cyan

    $vr = Invoke-Match -TestElo $candidateElo -NumGames $VerifyGames -Label "verify_r$v" -MoveTime $VerifyMoveTime
    if (-not $vr) { try { Stop-Transcript | Out-Null } catch {}; exit 1 }

    $verrStr = if ($vr.EloError) { "+/- $($vr.EloError)" } else { "+/- n/a" }

    Write-Host ""
    Write-Host "  Result    : $($vr.Wins)W / $($vr.Losses)L / $($vr.Draws)D | Score: $($vr.Score)" -ForegroundColor Green
    Write-Host "  CLI diff  : $($vr.EloDiff) $verrStr" -ForegroundColor Green

    if ($null -ne $vr.EloDiff) {
        $allEloDiffs += $vr.EloDiff
        if ([Math]::Abs($vr.EloDiff) -gt $VerifyEloBand) {
            $verified = $false
            Write-Host "  >> Diff $($vr.EloDiff) outside band +/-$VerifyEloBand -- estimate may be rough." -ForegroundColor Yellow
        } else {
            Write-Host "  >> Within verification band." -ForegroundColor Green
        }
    } else {
        $verified = $false
        Write-Host "  >> Elo diff unavailable for this round -- marking as unverified." -ForegroundColor Yellow
    }
}

# ============================================================
# FINAL REPORT
# ============================================================
$avgEloDiff = if ($allEloDiffs.Count -gt 0) {
    ($allEloDiffs | Measure-Object -Average).Average
} else { 0 }

$finalElo = Limit-Elo ([int][Math]::Round($candidateElo + $avgEloDiff))

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  FINAL RESULT"                                                    -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan

# FIX #17: Save verified Elo to confirmed file, unverified to estimate file
if ($verified) {
    Write-Host "  Status          : VERIFIED" -ForegroundColor Green
    # FIX 4.23: Use .NET WriteAllText to avoid UTF-8 BOM that PS 5.1 Out-File emits.
    [System.IO.File]::WriteAllText($EloFile, ($finalElo | Out-String).Trim(), [System.Text.UTF8Encoding]::new($false))
    Write-Host "  Saved to        : $EloFile" -ForegroundColor Gray
} else {
    Write-Host "  Status          : BEST ESTIMATE (verification showed variance)" -ForegroundColor Yellow
    [System.IO.File]::WriteAllText($EstimateEloFile, ($finalElo | Out-String).Trim(), [System.Text.UTF8Encoding]::new($false))
    Write-Host "  Saved to        : $EstimateEloFile (estimate only -- not overwriting confirmed Elo)" -ForegroundColor Yellow
}

Write-Host "  Estimated Elo   : ~$finalElo"                                                                         -ForegroundColor White
Write-Host "  Avg verify diff : $([Math]::Round($avgEloDiff,1)) (applied to refine final Elo)"                      -ForegroundColor White
Write-Host "  Confidence      : $VerifyGames x $VerifyRounds = $($VerifyGames*$VerifyRounds) games"                 -ForegroundColor White
Write-Host "  Cal time/move   : ${CalMoveTime}s  |  Verify time/move: ${VerifyMoveTime}s  |  Probe: ${ProbeMoveTimeMs}ms  |  Concurrency: $Concurrency" -ForegroundColor White
Write-Host "  Log file        : $LogFile"                                                                            -ForegroundColor White
Write-Host "================================================================" -ForegroundColor Cyan

# FIX #2: Cleanup PGN files only if -KeepPGN is not set
if (-not $KeepPGN) {
    Get-ChildItem -Path $ReleaseDir -Filter "elo_cal_*.pgn" -ErrorAction SilentlyContinue | Remove-Item -Force
    Write-Host "  Cleaned up calibration PGN files." -ForegroundColor DarkGray
} else {
    $pgnFiles = Get-ChildItem -Path $ReleaseDir -Filter "elo_cal_*.pgn" -ErrorAction SilentlyContinue
    if ($pgnFiles) {
        Write-Host "  PGN files retained (-KeepPGN):" -ForegroundColor DarkGray
        $pgnFiles | ForEach-Object { Write-Host "    $($_.FullName)" -ForegroundColor DarkGray }
    }
}

# FIX #16: Stop transcript
try { Stop-Transcript | Out-Null } catch {}

# FIX #18: Meaningful exit codes (0 = verified, 4 = unverified but completed)
if ($verified) {
    exit 0
} else {
    exit 4
}
