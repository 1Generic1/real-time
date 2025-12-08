@echo off
color 07
cls

echo.
echo ========================================
echo.

rem Initial animation
echo Starting up.
timeout /t 0.3 >nul
cls
echo.
echo ========================================
echo.
echo Starting up..
timeout /t 0.3 >nul
cls
echo.
echo ========================================
echo.
echo Starting up...
timeout /t 0.3 >nul
cls

echo.
echo ========================================
echo.

rem Typewriter effect
set "text=LAUNCHING TRADING SYSTEM"
setlocal enabledelayedexpansion
for /L %%i in (0,1,22) do (
    set "char=!text:~%%i,1!"
    <nul set /p "=!char!"
    timeout /t 0.05 >nul
)
endlocal

echo.
echo.

rem Progress animation
for %%a in ("⠋" "⠙" "⠹" "⠸" "⠼" "⠴" "⠦" "⠧" "⠇" "⠏") do (
    <nul set /p "=Initializing [%%~a]"
    timeout /t 0.1 >nul
    <nul set /p "=%BS%%BS%%BS%%BS%%BS%%BS%%BS%%BS%%BS%%BS%%BS%%BS%%BS%"
)
echo Initializing [✓]

echo.
echo ========================================
echo.
echo ✅ SYSTEM READY
echo.
timeout /t 1 >nul

cd /d "C:\Users\DELL\source\realtime\real-time\crypto-signal-dashboard"
python trading_execution_systemsimple7.py

echo.
echo ========================================
echo          END OF SESSION
echo ========================================
echo.
pause