FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04

RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3-pip \
    ffmpeg \
    git \

    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages \
    torch torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY server.py .
COPY app/ ./app/

EXPOSE 8080 8765

CMD ["python", "server.py"]

