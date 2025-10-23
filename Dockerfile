# Use Python 3.9 as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install Python dependencies
COPY backend/requirements.txt /app/backend/
WORKDIR /app/backend
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ /app/backend/
COPY data/ /app/data/

# Change back to app root
WORKDIR /app

# Install Node.js for frontend
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Copy frontend package files
COPY package*.json ./
RUN npm install

# Copy frontend source code
COPY public/ ./public/
COPY src/ ./src/

# Build frontend
RUN npm run build

# Copy build to a location that can be served
RUN cp -r build/* /app/backend/static/

# Expose port
EXPOSE 8000

# Set working directory to backend
WORKDIR /app/backend

# Start the backend server (which can also serve static files)
CMD ["python", "main.py"]
