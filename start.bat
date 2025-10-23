@echo off
echo Starting Boosting Algorithms Demo...

REM Check if Docker is available
docker-compose --version >nul 2>&1
if %errorlevel% == 0 (
    echo Using Docker Compose...
    docker-compose up
    goto end
)

docker --version >nul 2>&1
if %errorlevel% == 0 (
    echo Using Docker...
    docker build -t boosting-demo .
    docker run -p 8000:8000 boosting-demo
    goto end
)

echo Docker not found. Starting manually...

REM Start backend
echo Starting backend server...
cd backend
start "Backend Server" python main.py

REM Wait for backend to start
timeout /t 3 /nobreak >nul

REM Back to root directory
cd ..

REM Start frontend (if in development mode)
if exist package.json (
    echo Starting frontend development server...
    start "Frontend Server" npm start
)

echo.
echo Application started!
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000 (if npm start was used)
echo.
echo Press any key to stop this script (services will continue running)
pause >nul

:end
