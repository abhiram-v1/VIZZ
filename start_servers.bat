@echo off
echo ============================================================
echo 🚀 BOOSTING ALGORITHMS DEMO - SERVER STARTUP
echo ============================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python first.
    pause
    exit /b 1
)

REM Check if Node.js is available
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js not found! Please install Node.js first.
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "package.json" (
    echo ❌ package.json not found! Please run from project root.
    pause
    exit /b 1
)

if not exist "backend" (
    echo ❌ backend folder not found! Please run from project root.
    pause
    exit /b 1
)

echo ✅ Environment check passed!
echo.

REM Start the Python script
echo 🚀 Starting servers using Python script...
python start_servers.py

REM If we get here, the script finished
echo.
echo 👋 Servers stopped. Press any key to exit...
pause >nul
