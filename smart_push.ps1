# smart_push.ps1
# Force pushes to origin/main, automatically skipping files over 100MB.
# .bin files are excluded as a pattern (*.bin) rather than individually.
$ErrorActionPreference = "Stop"  # F6.2: fail fast on errors

$maxBytes = 100MB
$gitignorePath = ".gitignore"

# --- Step 1: Add *.bin pattern to .gitignore if not already present ---
$gitignoreContent = if (Test-Path $gitignorePath) { Get-Content $gitignorePath -Raw } else { "" }

if ($gitignoreContent -notmatch '(?m)^\*\.bin$') {
    Write-Host "Adding *.bin to .gitignore..."
    Add-Content $gitignorePath "`n# Large binary training/weight files`n*.bin"
} else {
    Write-Host "*.bin already in .gitignore, skipping..."
}

# Unstage all .bin files from git tracking
Write-Host "Removing *.bin files from git tracking..."
git rm --cached -r --ignore-unmatch "*.bin" 2>$null

# WARNING: File paths are not validated against special characters (backticks, $, quotes).
# Consider using git lfs for large binary management instead of force-push exclusion.

# FIX 12.6: Re-read .gitignore after *.bin mutation to avoid stale checks
$gitignoreContent = if (Test-Path $gitignorePath) { Get-Content $gitignorePath -Raw } else { "" }

# --- Step 2: Find and skip any other files over 100MB (non-.bin) ---
Write-Host "Scanning for other oversized files (>100MB)..."
$foundLarge = $false

$largeFiles = Get-ChildItem -Recurse -File | Where-Object {
    $_.Length -gt $maxBytes -and $_.Extension -ne ".bin"
}
foreach ($file in $largeFiles) {
    $foundLarge = $true
    # NOTE: Assumes $file.FullName starts with (Get-Location).Path (i.e., file is under git root).
    # Fails on symlinks, junctions, or if Get-Location has a trailing separator.
    $rel = $file.FullName.Substring((Get-Location).Path.Length + 1).Replace('\', '/')
    Write-Host "Skipping (too large): $rel ($([math]::Round($file.Length / 1MB, 1)) MB)"

    if ($gitignoreContent -notmatch [regex]::Escape($rel)) {
        Add-Content $gitignorePath "`n$rel"
    }

    git rm --cached $rel 2>$null
}

if (-not $foundLarge) {
    Write-Host "No other oversized files found."
}

# --- Step 3: Stage .gitignore, commit if there are changes, then force push ---
git add .gitignore

$staged = git diff --cached --name-only
if ($staged) {
    Write-Host "Committing .gitignore changes..."
    git commit -m "chore: exclude .bin and other large files from tracking"
}

# Safety: verify we're on main and get confirmation
$currentBranch = git rev-parse --abbrev-ref HEAD
if ($currentBranch -ne "main") {
    Write-Error "Not on main branch (current: $currentBranch). Aborting."
    exit 1
}

$confirm = Read-Host "Force push to origin/main? This rewrites remote history. (y/N)"
if ($confirm -ne "y") {
    Write-Host "Aborted."
    exit 0
}

Write-Host "Force pushing to origin/main (with lease)..."
git push origin main --force-with-lease
if ($LASTEXITCODE -ne 0) {
    Write-Error "Push failed with exit code $LASTEXITCODE"
    exit 1
}

Write-Host ""
Write-Host "Done! .bin files are excluded by pattern and will be ignored in all future pushes too."
