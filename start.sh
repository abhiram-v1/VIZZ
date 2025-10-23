#!/bin/bash

# Start the Boosting Algorithms Demo

echo "Starting Boosting Algorithms Demo..."

# Check if Docker is available
if command -v docker-compose &> /dev/null; then
    echo "Using Docker Compose..."
    docker-compose up
elif command -v docker &> /dev/null; then
    echo "Using Docker..."
    docker build -t boosting-demo .
    docker run -p 8000:8000 boosting-demo
else
    echo "Docker not found. Starting manually..."
    
    # Start backend
    echo "Starting backend server..."
    cd backend
    python main.py &
    BACKEND_PID=$!
    
    # Wait a moment for backend to start
    sleep 3
    
    # Start frontend (if in development mode)
    cd ..
    if [ -f "package.json" ]; then
        echo "Starting frontend development server..."
        npm start &
        FRONTEND_PID=$!
    fi
    
    echo "Application started!"
    echo "Backend: http://localhost:8000"
    echo "Frontend: http://localhost:3000 (if npm start was used)"
    
    # Wait for user to stop
    echo "Press Ctrl+C to stop all services"
    wait
fi
