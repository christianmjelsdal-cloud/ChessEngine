@echo off
rem INFO [13.12]: Set CWD to script dir for relative asset paths
cd /d "%~dp0"
"%~dp0x64\Release\ChessEngine.exe" --uci
