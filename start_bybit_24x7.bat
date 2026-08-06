@echo off
REM Launch the Bybit 24/7 keep-alive loop.
REM %~dp0 = this file's folder. The single quotes around the path matter: without
REM them PowerShell splits on the space in "Trader Bot" and refuses the script.
powershell -NoProfile -ExecutionPolicy Bypass -Command "& '%~dp0run_bybit_24x7.ps1'"
