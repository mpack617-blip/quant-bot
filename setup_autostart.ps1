# Registers the 24/7 launcher to run automatically at every Windows logon, so the
# bot survives PC restarts. Run ONCE:
#   powershell -ExecutionPolicy Bypass -File "D:\Trader Bot\quant-bot\setup_autostart.ps1"
# Remove later with:  Unregister-ScheduledTask -TaskName "QuantBot24x7" -Confirm:$false

$Launcher = "D:\Trader Bot\quant-bot\start_24x7.ps1"
$TaskName = "QuantBot24x7"

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-WindowStyle Hidden -ExecutionPolicy Bypass -File `"$Launcher`""
$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Settings $settings -Description "Quant-Bot 24/7 trading loop + cockpit" -Force | Out-Null

Write-Host "Registered scheduled task '$TaskName' (runs at logon)."
Write-Host "Test it now without rebooting:  Start-ScheduledTask -TaskName $TaskName"
Write-Host "To stop autostart:              Unregister-ScheduledTask -TaskName $TaskName -Confirm:`$false"
Write-Host ""
Write-Host "IMPORTANT: keep the PC powered on and prevent sleep for true 24/7:"
Write-Host '  powercfg /change standby-timeout-ac 0'
