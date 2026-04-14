@echo off
REM ============================================================================
REM  Build and run the Chess Engine Test Suite
REM
REM  Usage:
REM    build_tests.bat              Build + run (Release x64)
REM    build_tests.bat debug        Build + run (Debug x64)
REM    build_tests.bat build        Build only (Release x64)
REM    build_tests.bat --verbose    Build + run with verbose output
REM
REM  Prerequisites:
REM    - Visual Studio 2022 with C++ workload installed
REM    - Run from a "Developer Command Prompt" or "x64 Native Tools" prompt
REM      OR the script will try to find vcvarsall.bat automatically
REM ============================================================================

setlocal enabledelayedexpansion

set CONFIG=Release
set RUNARGS=
set BUILDONLY=0

:parse_args
if "%~1"=="" goto done_args
if /i "%~1"=="debug" set CONFIG=Debug& shift & goto parse_args
if /i "%~1"=="build" set BUILDONLY=1& shift & goto parse_args
if "%~1"=="--verbose" set RUNARGS=--verbose& shift & goto parse_args
if "%~1"=="-v" set RUNARGS=--verbose& shift & goto parse_args
echo [WARNING] Unknown argument: %~1 (ignored)
echo          Valid arguments: debug, build, --verbose, -v
shift
goto parse_args
:done_args

REM --- Try to set up VS environment if cl.exe is not available ---
where cl.exe >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] cl.exe not found, searching for Visual Studio...
    set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
    if exist "!VSWHERE!" (
        for /f "usebackq tokens=*" %%i in (`"!VSWHERE!" -latest -property installationPath`) do set "VSINSTALL=%%i"
        if exist "!VSINSTALL!\VC\Auxiliary\Build\vcvarsall.bat" (
            echo [INFO] Found VS at: !VSINSTALL!
            call "!VSINSTALL!\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
        )
    )
)

where cl.exe >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] cl.exe not found. Please run from a Developer Command Prompt.
    exit /b 1
)

echo ============================================
echo  Building TestSuite (%CONFIG% x64)
echo ============================================

REM --- Compile all source files ---
set SOURCES=TestSuite.cpp Board.cpp Engine.cpp MoveGen.cpp NNUE.cpp NNUETrainer.cpp SelfPlayGen.cpp UCI.cpp
set CFLAGS=/EHsc /std:c++17 /W3 /DTEST_BUILD /D_CONSOLE

if /i "%CONFIG%"=="Debug" (
    set CFLAGS=%CFLAGS% /Zi /Od /D_DEBUG /MDd
) else (
    set CFLAGS=%CFLAGS% /O2 /DNDEBUG /MD
)

if not exist "build_test" mkdir build_test

cl %CFLAGS% %SOURCES% /Fe:build_test\TestSuite.exe /Fo:build_test\ /link /SUBSYSTEM:CONSOLE
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Build failed!
    exit /b 1
)

echo.
echo [OK] Build successful: build_test\TestSuite.exe
echo.

if %BUILDONLY%==1 (
    echo [INFO] Build-only mode, skipping test run.
    exit /b 0
)

echo ============================================
echo  Running Tests
echo ============================================
echo.

build_test\TestSuite.exe %RUNARGS%
set RESULT=%ERRORLEVEL%

echo.
if %RESULT%==0 (
    echo [OK] All tests passed!
) else (
    echo [FAIL] Some tests failed (exit code: %RESULT%)
)

exit /b %RESULT%
