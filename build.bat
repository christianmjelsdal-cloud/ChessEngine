@echo off
msbuild ChessEngine.sln /t:Rebuild /p:Configuration=Release;Platform=x64 /fileLogger /fileLoggerParameters:LogFile=build.txt;Verbosity=normal;Encoding=UTF-8
