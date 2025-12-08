@echo off
chcp 65001 >nul
title 🚀 Trading System Launcher
color 0A

echo.
echo ╔══════════════════════════════════════════════════════╗
echo ║               TRADING SYSTEM LAUNCHER                ║
echo ╚══════════════════════════════════════════════════════╝
echo.
echo [1] 🚀 Quick Start (BTC/USDT 4H)
echo [2] 📊 Batch Analysis (Multiple Cryptos)
echo [3] ⚙️  Run All Tests
echo [4] 🔧 Run with Custom Parameters
echo [5] ❌ Exit
echo.

set /p choice="Select option (1-5): "

if "%choice%"=="1" (
    echo.
    echo 🚀 Starting BTC/USDT analysis...
    timeout /t 1 /nobreak >nul
    python trading_execution_systemsimple7.py
    goto end
)

if "%choice%"=="2" (
    echo.
    echo 📊 Starting batch analysis...
    timeout /t 1 /nobreak >nul
    python -c "from trading_execution_systemsimple7 import batch_analyze_cryptos; batch_analyze_cryptos(['BTC/USDT', 'ETH/USDT', 'SOL/USDT'], enable_ml=True)"
    goto end
)

if "%choice%"=="3" (
    echo.
    echo ⚙️ Running all system tests...
    timeout /t 1 /nobreak >nul
    python trading_execution_systemsimple7.py
    goto end
)

if "%choice%"=="4" (
    echo.
    set /p symbol="Enter symbol (e.g., BTC/USDT): "
    set /p tf="Enter timeframe (4h, 1h, 1d): "
    echo.
    echo 🔧 Running custom analysis for %symbol% on %tf% timeframe...
    timeout /t 2 /nobreak >nul
    python -c "from trading_execution_systemsimple7 import integrate_and_trade_with_ml; integrate_and_trade_with_ml(symbol='%symbol%', timeframe='%tf%', account_balance=1000, enable_ml=True)"
    goto end
)

if "%choice%"=="5" (
    echo.
    echo 👋 Exiting...
    timeout /t 1 /nobreak >nul
    exit
)

echo.
echo ❌ Invalid choice!
timeout /t 2 /nobreak >nul

:end
echo.
echo ========================================
echo ✅ Execution complete!
echo ========================================
echo.
pause