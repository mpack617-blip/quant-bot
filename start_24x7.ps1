# Quant-Bot 24/7 launcher (Windows).
#  - Makes sure TradingView is up with the CDP debug port (9222) -- WITHOUT killing a
#    healthy instance (a slow CDP response no longer triggers a restart).
#  - Starts the cockpit (which auto-starts the trading loop AND auto-opens the browser)
#    and keeps it alive, restarting it if it ever exits.
#
# Launch it via the wrapper so the space in "Trader Bot" can't break the path:
#     "D:\Trader Bot\quant-bot\start_24x7.bat"     (double-click, or point Task Scheduler here)

$ErrorActionPreference = "SilentlyContinue"
$BotDir = "D:\Trader Bot\quant-bot"
$Port = 9222
$Python = "C:\Users\chaud\AppData\Local\Programs\Python\Python314\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }   # fall back to PATH

# Lenient CDP check: 5s timeout, a few retries, so a slow first response is not a false "down".
function Test-CDP {
    for ($t = 0; $t -lt 3; $t++) {
        try { Invoke-RestMethod -Uri "http://localhost:$Port/json/version" -TimeoutSec 5 -ErrorAction Stop | Out-Null; return $true }
        catch { Start-Sleep -Seconds 2 }
    }
    return $false
}

function Start-TradingView {
    $pkg = Get-AppxPackage *TradingView* | Select-Object -First 1
    if ($pkg) {
        $exe = Join-Path $pkg.InstallLocation "TradingView.exe"
        if (Test-Path $exe) { Start-Process $exe -ArgumentList "--remote-debugging-port=$Port"; return }
        Write-Host "TradingView.exe not found in $($pkg.InstallLocation)"
    } else { Write-Host "TradingView appx package not found." }
}

# --- 1. Ensure TradingView/CDP is up. Only launch if it is genuinely down. Never kill a live one. ---
if (Test-CDP) {
    Write-Host "$(Get-Date -Format 'u')  CDP already up -- leaving TradingView as is."
} else {
    Write-Host "$(Get-Date -Format 'u')  CDP down. Launching TradingView with debug port $Port ..."
    if (-not (Get-Process TradingView -ErrorAction SilentlyContinue)) {
        Start-TradingView
    } else {
        # A TradingView is running but without the debug port; restart it WITH the port.
        Get-Process TradingView -ErrorAction SilentlyContinue | Stop-Process -Force
        Start-Sleep -Seconds 2
        Start-TradingView
    }
    Write-Host "Waiting for CDP to come up (can take ~30s)..."
    for ($i = 0; $i -lt 40; $i++) {
        if (Test-CDP) { Write-Host "CDP is up."; break }
        Start-Sleep -Seconds 2
    }
}
if (-not (Test-CDP)) { Write-Host "WARNING: CDP still down -- the bot will retry; orders may fail until TradingView is ready." }

# --- 2. Keep the cockpit alive forever ---
Set-Location $BotDir
$env:QUANT_MODE = "tradingview"
$env:QUANT_AUTOSTART = "1"
while ($true) {
    Write-Host "$(Get-Date -Format 'u')  starting cockpit (TradingView mode)..."
    & $Python "$BotDir\cockpit.py"
    Write-Host "$(Get-Date -Format 'u')  cockpit exited -- restarting in 10s (CDP check first)..."
    if (-not (Test-CDP)) {
        if (-not (Get-Process TradingView -ErrorAction SilentlyContinue)) { Start-TradingView }
        Start-Sleep -Seconds 20
    }
    Start-Sleep -Seconds 10
}
