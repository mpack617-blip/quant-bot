# ============================================================
#  Quant-Bot 24/7 keep-alive — BYBIT mode.
#
#  Simpler than the TradingView launcher: no chart, no CDP port, no browser to
#  babysit. Orders go straight to the exchange over REST, so the only thing that
#  has to stay alive is this python process.
#
#  Loop: start the cockpit, wait for it to exit, restart it. A crash at 3am
#  therefore costs one tick, not the night.
# ============================================================

$ErrorActionPreference = "Continue"
$Root   = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = "C:\Users\chaud\AppData\Local\Programs\Python\Python314\python.exe"
$Log    = Join-Path $Root "_bybit_24x7.log"

function Say($msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg"
    Write-Output $line
    Add-Content -Path $Log -Value $line -Encoding utf8
}

# The bot trades a NOTIONAL book of this size, not the demo account's balance.
# Change QUANT_CAPITAL to rehearse a different account size.
$env:QUANT_MODE      = "bybit"
$env:QUANT_CAPITAL   = "200"
$env:QUANT_PERIOD    = "180"
$env:QUANT_AUTOSTART = "1"
$env:QUANT_NOBROWSER = "1"   # a 24/7 background service must not pop browser windows

Say "=== Bybit 24/7 launcher started (capital `$$($env:QUANT_CAPITAL), tick $($env:QUANT_PERIOD)s) ==="

if (-not (Test-Path $Python)) { Say "FATAL: python not found at $Python"; exit 1 }

# Don't fight a cockpit that is already serving — otherwise two runners share one
# journal.db and one state file and quietly corrupt each other's books.
try {
    $r = Invoke-WebRequest -Uri "http://localhost:8787/api/status" -TimeoutSec 5 -UseBasicParsing
    if ($r.StatusCode -eq 200) { Say "cockpit already running on 8787 - nothing to do, exiting."; exit 0 }
} catch { }

$fails = 0
while ($true) {
    Say "starting cockpit..."
    & $Python (Join-Path $Root "cockpit.py") 2>&1 | ForEach-Object { Add-Content -Path $Log -Value $_ -Encoding utf8 }
    $code = $LASTEXITCODE
    Say "cockpit exited (code $code)"

    # Back off if it dies immediately and repeatedly — a tight restart loop on a
    # real fault just fills the disk with logs and hides the actual error.
    $fails++
    if ($fails -ge 5) { Say "5 consecutive exits - backing off 5 minutes"; Start-Sleep -Seconds 300; $fails = 0 }
    else { Start-Sleep -Seconds 10 }
}
