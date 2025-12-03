# Use latest CUDA 13 with cuDNN
FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04

# Install Python 3.12 and dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3

# Set CUDA paths
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

WORKDIR /app

# Install dependencies first (for caching)
COPY requirements.txt .

# Install PyTorch with CUDA 12.4 support (compatible with CUDA 13)
# PyTorch doesn't have cu13 wheels yet, cu124 is the latest and works with CUDA 13
RUN pip install --no-cache-dir --break-system-packages \
    torch torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir --break-system-packages -r requirements.txt

# Copy model download script
COPY scripts/download_models.py ./scripts/

# Build arguments for model configuration
ARG MODEL_NAME=large-v3
ARG HF_TOKEN=""

# Set environment variables for model download
ENV MODEL_NAME=${MODEL_NAME}
ENV MODEL_PATH=/app/models
ENV HF_TOKEN=${HF_TOKEN}

# Create cache directories
RUN mkdir -p /app/models /root/.cache/torch/hub /root/.cache/huggingface

# Pre-download models during build
# This caches models in the image so they don't need to be downloaded at runtime
RUN python scripts/download_models.py

# Copy application code
COPY server.py .
COPY app/ ./app/

# Ensure MODEL_PATH is set for runtime
ENV MODEL_PATH=/app/models

EXPOSE 8080 8765

CMD ["python", "server.py"]
