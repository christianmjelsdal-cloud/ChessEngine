@echo off
setlocal enabledelayedexpansion

REM === Chess Engine NNUE Training Pipeline ===
REM Usage:
REM   train.bat                    Quick training run
REM   train.bat --full             Full training run
REM   train.bat --quick            Quick training run (explicit)
REM   train.bat [custom args]      Pass args to TrainingRunner.exe

set PRESET=--quick
if "%~1" neq "" (
    REM FIX 12.1: Validate PRESET against allowlist to prevent command injection via %*
    set "VALID_PRESET="
    for %%P in (--quick --full --benchmark --resume --help) do (
        if /I "%~1"=="%%P" set "VALID_PRESET=1"
    )
    if defined VALID_PRESET (
        set "PRESET=%~1"
    ) else (
        echo [ERROR] Unknown preset: %~1
        echo         Valid presets: --quick --full --benchmark --resume --help
        exit /b 1
    )
)

echo.
echo ============================================
echo   Chess Engine NNUE Training Pipeline
echo ============================================
echo.

REM Check for Python (needed for dashboard)
set NO_DASHBOARD=
where python >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Python not found. Dashboard will not be available.
    echo           Install Python 3 from python.org for live dashboard.
    set NO_DASHBOARD=1
)

REM Try to find Visual Studio
REM NOTE: VS discovery uses hardcoded paths. Consider using vswhere.exe for reliable detection.
REM See build_tests.bat for the vswhere approach.
set VCVARS=
for %%v in (2022 2019 2017) do (
    for %%e in (Enterprise Professional Community BuildTools) do (
        if exist "C:\Program Files\Microsoft Visual Studio\%%v\%%e\VC\Auxiliary\Build\vcvars64.bat" (
            set "VCVARS=C:\Program Files\Microsoft Visual Studio\%%v\%%e\VC\Auxiliary\Build\vcvars64.bat"
            goto :found_vc
        )
        if exist "C:\Program Files (x86)\Microsoft Visual Studio\%%v\%%e\VC\Auxiliary\Build\vcvars64.bat" (
            set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\%%v\%%e\VC\Auxiliary\Build\vcvars64.bat"
            goto :found_vc
        )
    )
)

:found_vc
if "!VCVARS!" == "" (
    echo [ERROR] Visual Studio not found. Please install Visual Studio with C++ tools.
    echo         Or compile manually:
    echo         cl /EHsc /std:c++17 /O2 /DNDEBUG /DNOMINMAX TrainingRunner.cpp Board.cpp Engine.cpp MoveGen.cpp NNUE.cpp NNUETrainer.cpp SelfPlayGen.cpp /Fe:TrainingRunner.exe
    exit /b 1
)

echo [1/3] Setting up build environment...
call "!VCVARS!" >nul 2>&1

REM Build TrainingRunner.exe
echo [2/3] Building TrainingRunner.exe...
set SRC=TrainingRunner.cpp Board.cpp Engine.cpp MoveGen.cpp NNUE.cpp NNUETrainer.cpp SelfPlayGen.cpp
set CFLAGS=/EHsc /std:c++17 /O2 /DNDEBUG /DNOMINMAX /D_CRT_SECURE_NO_WARNINGS /W3
cl %CFLAGS% %SRC% /Fe:TrainingRunner.exe /link /MACHINE:X64 >build_training.log 2>&1

if errorlevel 1 (
    echo [ERROR] Build failed! Check build_training.log for details.
    echo.
    echo --- Build Output ---
    type build_training.log
    echo --- End Build Output ---
    exit /b 1
)

echo        Build successful!

REM Create assets directory if needed
if not exist "assets" mkdir assets

REM Launch dashboard in background
REM WARNING: Dashboard binds to 0.0.0.0:8080 with no authentication.
REM On shared systems or cloud VMs, this exposes training status to the network.
REM Consider binding to 127.0.0.1 (--host 127.0.0.1) if on a shared network.
if not defined NO_DASHBOARD (
    echo [3/3] Launching dashboard + training...
    start "Training Dashboard" python dashboard\server.py training_log.jsonl 8080
    timeout /t 2 /nobreak >nul
) else (
    echo [3/3] Starting training (no dashboard^)...
)

REM Run training
echo.
echo Starting training with: %PRESET%
echo ──────────────────────────────────────────
TrainingRunner.exe %PRESET%

echo.
echo ──────────────────────────────────────────
echo Training complete!
if not defined NO_DASHBOARD (
    echo Dashboard is still running at http://localhost:8080
    echo Press any key to stop the dashboard...
    pause >nul
    REM FIX 12.16: Use /F (force) /T (kill child tree) for reliable cleanup.
    REM Window title matching is fragile if the title changes at runtime.
    taskkill /F /T /FI "WINDOWTITLE eq Training Dashboard" >nul 2>&1
)

endlocal
