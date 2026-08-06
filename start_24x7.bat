@echo off
REM Robust 24/7 launcher for the Quant-Bot cockpit.
REM Double-click this, or point Task Scheduler at it. Handles the space in "Trader Bot".
REM %~dp0 = this file's folder (with trailing backslash); single quotes keep the space safe.
powershell -NoProfile -ExecutionPolicy Bypass -Command "& '%~dp0start_24x7.ps1'" > "%~dp0_bat_run.log" 2>&1
