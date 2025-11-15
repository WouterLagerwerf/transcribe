FROM python:3.11-slim

# Install system dependencies required for Whisper (ffmpeg) and torch
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY server.py .
COPY app/ ./app/

# Expose port
# 8080 for HTTP API server
EXPOSE 8080

# Set default environment variables (can be overridden at runtime)
ENV SERVER_HOST=0.0.0.0
ENV HEALTH_CHECK_PORT=8080
ENV MODEL_NAME=base
ENV MODEL_PATH=/app/models
ENV PROCESSING_THREADS=4

# Create directory for models
RUN mkdir -p /app/models

# Run the server
CMD ["python", "server.py"]

