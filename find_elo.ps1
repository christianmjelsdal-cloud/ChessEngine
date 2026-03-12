# ============================================================
# find_elo.ps1 - Auto-calibrate engine Elo vs Stockfish
# ============================================================
# Usage: .\find_elo.ps1
# ============================================================

# ---- CONFIG ------------------------------------------------
$MyEngineCmd  = "x64\Release\ChessEngine.exe"
$StockfishCmd = "stockfish\stockfish.exe"
$CutechessCmd = "cutechess\cutechess-cli.exe"

$SearchGames  = 50      # Games per round during search
$VerifyGames  = 100     # Games per verification round
$VerifyRounds = 2       # Number of verification rounds after convergence
$SearchDepth  = 8       # Fixed depth for both engines
$MoveTime     = 30      # Max seconds per move (safety net)

$DefaultElo   = 2500    # Starting assumption if no prior result
$EloFile      = "last_confirmed_elo.txt"  # Persists last verified Elo

$MaxIterations = 10     # Safety cap on search iterations
$ConvergeThreshold = 15 # Stop searching when Elo adjustment < this
$VerifyEloBand = 30     # Verification passes if |Elo diff| < this
# ------------------------------------------------------------

function Run-Match {
    param([int]$TestElo, [int]$NumGames, [string]$Label)

    $halfRounds = [Math]::Max(1, [int]($NumGames / 2))

    $cutechessArgs = @(
        "-engine", "name=MyEngine", "cmd=$MyEngineCmd", "arg=--uci",
        "-engine", "name=Stockfish", "cmd=$StockfishCmd",
            "option.UCI_LimitStrength=true", "option.UCI_Elo=$TestElo",
        "-each", "proto=uci", "depth=$SearchDepth", "st=$MoveTime",
        "-rounds", "$halfRounds", "-games", "2", "-repeat",
        "-recover",
        "-pgnout", "elo_cal_${Label}.pgn"
    )

    $output = & $CutechessCmd @cutechessArgs 2>&1
    $output | ForEach-Object { Write-Host $_ }

    # Parse score line
    $scoreLine = $output | Where-Object { $_ -match "^Score of MyEngine vs Stockfish:" } | Select-Object -Last 1
    if (-not $scoreLine) {
        Write-Host "ERROR: Could not parse score line." -ForegroundColor Red
        return $null
    }

    $wins = 0; $losses = 0; $draws = 0; $score = 0.0
    if ($scoreLine -match "(\d+)\s*-\s*(\d+)\s*-\s*(\d+)\s*\[([0-9.]+)\]") {
        $wins   = [int]$Matches[1]
        $losses = [int]$Matches[2]
        $draws  = [int]$Matches[3]
        $score  = [double]$Matches[4]
    }

    # Parse Elo difference line from cutechess-cli
    # Format: "Elo difference: -50.2 +/- 35.1, LOS: 15.2 %, DrawRatio: 30.0 %"
    $eloDiffValue = 0.0
    $eloError = 0.0
    $eloLine = $output | Where-Object { $_ -match "Elo difference:" } | Select-Object -Last 1
    if ($eloLine -match "Elo difference:\s*([-0-9.]+)\s*\+/-\s*([0-9.]+)") {
        $eloDiffValue = [double]$Matches[1]
        $eloError = [double]$Matches[2]
    }

    return @{
        Wins     = $wins
        Losses   = $losses
        Draws    = $draws
        Score    = $score
        Total    = $wins + $losses + $draws
        EloDiff  = $eloDiffValue   # Positive = MyEngine stronger than SF at this Elo
        EloError = $eloError       # +/- margin
    }
}

# ---- Determine starting Elo ----
$startElo = $DefaultElo
if (Test-Path $EloFile) {
    $lastElo = Get-Content $EloFile -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($lastElo -match "^\d+$") {
        $startElo = [int]$lastElo
        Write-Host "  Loaded last confirmed Elo: $startElo from $EloFile" -ForegroundColor Green
    }
}

$currentElo = $startElo

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  Engine Elo Calibration (Elo-difference method)" -ForegroundColor Cyan
Write-Host "  Starting Elo guess: $currentElo" -ForegroundColor Cyan
Write-Host "  Phase 1: Iterative search ($SearchGames games/round)" -ForegroundColor Cyan
Write-Host "  Phase 2: Verification ($VerifyGames games x $VerifyRounds rounds)" -ForegroundColor Cyan
Write-Host "  Converge when adjustment < $ConvergeThreshold Elo" -ForegroundColor Cyan
Write-Host "  Depth: $SearchDepth | Max seconds/move: $MoveTime" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan

# ---- PHASE 1: Iterative Elo-difference search ----
$candidateElo = $currentElo

for ($iteration = 1; $iteration -le $MaxIterations; $iteration++) {
    # Clamp to valid Stockfish UCI_Elo range
    $testElo = [Math]::Max(1000, [Math]::Min(3200, $candidateElo))

    Write-Host ""
    Write-Host "=== SEARCH Round $iteration | Testing SF Elo: $testElo ===" -ForegroundColor Yellow

    $result = Run-Match -TestElo $testElo -NumGames $SearchGames -Label "search_r$iteration"
    if (-not $result) { exit 1 }

    Write-Host ""
    Write-Host "  Result: $($result.Wins)W / $($result.Losses)L / $($result.Draws)D | Score: $($result.Score)" -ForegroundColor Green
    Write-Host "  Elo difference: $($result.EloDiff) +/- $($result.EloError)" -ForegroundColor Green

    # Apply the Elo difference as adjustment
    # If MyEngine is +80 Elo stronger than SF at $testElo, then engine ~= $testElo + 80
    $adjustment = [int][Math]::Round($result.EloDiff)

    if ([Math]::Abs($adjustment) -lt $ConvergeThreshold) {
        $candidateElo = $testElo
        Write-Host "  >> Converged! Adjustment ($adjustment) < threshold ($ConvergeThreshold)" -ForegroundColor Cyan
        Write-Host "  >> Candidate Elo: $candidateElo" -ForegroundColor Cyan
        break
    }

    # Move SF Elo toward where engine actually is
    $candidateElo = $testElo + $adjustment
    Write-Host "  Adjusting: $testElo + $adjustment = $candidateElo" -ForegroundColor Yellow

    # Clamp check
    if ($candidateElo -lt 1000) {
        Write-Host "  Engine appears below Elo 1000 (SF minimum). Setting to 1000." -ForegroundColor Yellow
        $candidateElo = 1000
        break
    }
    if ($candidateElo -gt 3200) {
        Write-Host "  Engine appears above Elo 3200 (SF maximum). Setting to 3200." -ForegroundColor Yellow
        $candidateElo = 3200
        break
    }
}

if ($iteration -gt $MaxIterations) {
    Write-Host "  Hit max iterations. Best estimate: $candidateElo" -ForegroundColor Yellow
}

# ---- PHASE 2: Verification ----
Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  VERIFICATION PHASE" -ForegroundColor Cyan
Write-Host "  Testing Elo $candidateElo with $VerifyGames games x $VerifyRounds rounds" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan

$allEloDiffs = @()
$verified = $true

for ($v = 1; $v -le $VerifyRounds; $v++) {
    Write-Host ""
    Write-Host "=== VERIFICATION Round $v ===" -ForegroundColor Cyan

    $result = Run-Match -TestElo $candidateElo -NumGames $VerifyGames -Label "verify_r$v"
    if (-not $result) { exit 1 }

    $allEloDiffs += $result.EloDiff

    Write-Host ""
    Write-Host "  Result: $($result.Wins)W / $($result.Losses)L / $($result.Draws)D | Score: $($result.Score)" -ForegroundColor Green
    Write-Host "  Elo difference: $($result.EloDiff) +/- $($result.EloError)" -ForegroundColor Green

    if ([Math]::Abs($result.EloDiff) -gt $VerifyEloBand) {
        $verified = $false
        Write-Host "  >> Elo diff $($result.EloDiff) exceeds band +/-$VerifyEloBand. Estimate may be rough." -ForegroundColor Yellow
    } else {
        Write-Host "  >> Within verification band. Looking good!" -ForegroundColor Green
    }
}

# ---- FINAL REPORT ----
$avgEloDiff = ($allEloDiffs | Measure-Object -Average).Average

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  FINAL RESULT" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan

if ($verified) {
    Write-Host "  Status         : VERIFIED" -ForegroundColor Green
    # Save confirmed Elo for next run
    $candidateElo | Out-File -FilePath $EloFile -Encoding utf8
    Write-Host "  Saved to       : $EloFile (will be used as starting point next run)" -ForegroundColor Gray
} else {
    Write-Host "  Status         : BEST ESTIMATE (verification showed variance)" -ForegroundColor Yellow
    # Still save it as a starting point - better than default
    $candidateElo | Out-File -FilePath $EloFile -Encoding utf8
}

Write-Host "  Estimated Elo  : ~$candidateElo" -ForegroundColor White
Write-Host "  Avg Elo diff   : $([Math]::Round($avgEloDiff, 1)) (from verification)" -ForegroundColor White
Write-Host "  Confidence     : $VerifyGames games x $VerifyRounds rounds = $($VerifyGames * $VerifyRounds) total verification games" -ForegroundColor White
Write-Host "  Search depth   : $SearchDepth" -ForegroundColor White
Write-Host "================================================================" -ForegroundColor Cyan
