# PowerShell script to start both servers
# Boosting Algorithms Demo - Server Startup

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "🚀 BOOSTING ALGORITHMS DEMO - SERVER STARTUP" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check if we're in the right directory
if (-not (Test-Path "package.json")) {
    Write-Host "❌ package.json not found! Please run from project root." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

if (-not (Test-Path "backend")) {
    Write-Host "❌ backend folder not found! Please run from project root." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Function to start backend
function Start-Backend {
    Write-Host "🚀 Starting Backend Server..." -ForegroundColor Yellow
    
    # Start backend in a new PowerShell window
    $backendScript = @"
cd backend
python main.py
"@
    
    Start-Process powershell -ArgumentList "-Command", $backendScript -WindowStyle Normal
    Start-Sleep -Seconds 3
    Write-Host "✅ Backend server started on http://localhost:8000" -ForegroundColor Green
}

# Function to start frontend
function Start-Frontend {
    Write-Host "🚀 Starting Frontend Server..." -ForegroundColor Yellow
    
    # Check if node_modules exists
    if (-not (Test-Path "node_modules")) {
        Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
        npm install
        if ($LASTEXITCODE -ne 0) {
            Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
            return $false
        }
        Write-Host "✅ Dependencies installed" -ForegroundColor Green
    }
    
    # Start frontend in a new PowerShell window
    $frontendScript = @"
npm start
"@
    
    Start-Process powershell -ArgumentList "-Command", $frontendScript -WindowStyle Normal
    Start-Sleep -Seconds 5
    Write-Host "✅ Frontend server started on http://localhost:3000" -ForegroundColor Green
    return $true
}

# Function to open browser
function Open-Browser {
    Write-Host "🌐 Opening browser..." -ForegroundColor Yellow
    Start-Sleep -Seconds 2
    Start-Process "http://localhost:3000"
    Write-Host "✅ Browser opened" -ForegroundColor Green
}

# Main execution
try {
    # Start backend
    Start-Backend
    
    # Start frontend
    if (Start-Frontend) {
        # Open browser
        Open-Browser
        
        Write-Host ""
        Write-Host "============================================================" -ForegroundColor Green
        Write-Host "✅ SERVERS RUNNING SUCCESSFULLY!" -ForegroundColor Green
        Write-Host "🌐 Frontend: http://localhost:3000" -ForegroundColor Cyan
        Write-Host "🔧 Backend: http://localhost:8000" -ForegroundColor Cyan
        Write-Host "📚 API Docs: http://localhost:8000/docs" -ForegroundColor Cyan
        Write-Host "============================================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "💡 Close the PowerShell windows to stop the servers" -ForegroundColor Yellow
        Write-Host "💡 Or press Ctrl+C in each server window" -ForegroundColor Yellow
        Write-Host ""
        
        # Keep this window open
        Read-Host "Press Enter to exit this launcher (servers will keep running)"
    } else {
        Write-Host "❌ Failed to start frontend" -ForegroundColor Red
        Write-Host "💡 Try running manually: npm start" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ Error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "💡 Try running the servers manually:" -ForegroundColor Yellow
    Write-Host "   Backend: cd backend && python main.py" -ForegroundColor White
    Write-Host "   Frontend: npm start" -ForegroundColor White
}

Read-Host "Press Enter to exit"
