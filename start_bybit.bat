@echo off
REM ============================================================
REM  Start the quant-bot trading on the BYBIT account.
REM
REM  Which account it hits is decided by "env" in bybit_keys.json:
REM      demo    = fake money, real prices   (start here)
REM      testnet = separate test site
REM      mainnet = REAL MONEY
REM
REM  Run `python setup_bybit.py` once first to configure and verify keys.
REM  Unlike TradingView mode this needs no chart, no CDP port, no browser
REM  automation - orders go straight to the exchange over the REST API.
REM ============================================================
cd /d "%~dp0"
set QUANT_MODE=bybit
set QUANT_PERIOD=180
"C:\Users\chaud\AppData\Local\Programs\Python\Python314\python.exe" cockpit.py
pause
