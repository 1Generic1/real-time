Clear-Host

# Just animate the main title
$text = "        LAUNCHING TRADING SYSTEM"
$border = "========================================"

Write-Host $border -ForegroundColor Cyan

Write-Host -NoNewline ">> "
foreach ($char in $text.ToCharArray()) {
    Write-Host -NoNewline $char -ForegroundColor Yellow
    Start-Sleep -Milliseconds 40
}
Write-Host ""

Write-Host $border -ForegroundColor Cyan
Write-Host ""

Write-Host "Initializing..." -ForegroundColor Yellow
Start-Sleep -Seconds 1

Set-Location "C:\Users\DELL\source\realtime\real-time\crypto-signal-dashboard"

python trading_execution_systemsimple8.py

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "         COMPLETE" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")