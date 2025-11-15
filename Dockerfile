FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Install Python 3.11
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python && \
    ln -s /usr/bin/python3.11 /usr/bin/python3

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install PyTorch with CUDA support first (before other dependencies)
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install remaining Python dependencies
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

