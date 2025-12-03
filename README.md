# 🎤 Real-Time Transcription Server

A high-performance, GPU-accelerated speech-to-text server with real-time speaker identification. Built with [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) and [Pyannote](https://github.com/pyannote/pyannote-audio).

## Features

- **Real-time streaming transcription** via WebSocket
- **Speaker identification** - automatically detect and label different speakers
- **Multi-tenant support** - handle multiple concurrent transcription sessions
- **GPU acceleration** - optimized for NVIDIA GPUs with CUDA
- **Pre-downloaded models** - fast container startup
- **HTTP API** - batch transcription endpoint for file uploads
- **Built-in VAD** - Voice Activity Detection filters silence automatically

---

## 📐 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRANSCRIPTION SERVER                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────────────┐ │
│  │   Client 1   │     │   Client 2   │     │   Client N                   │ │
│  │  (WebSocket) │     │  (WebSocket) │     │  (WebSocket)                 │ │
│  └──────┬───────┘     └──────┬───────┘     └──────────────┬───────────────┘ │
│         │                    │                            │                  │
│         └────────────────────┼────────────────────────────┘                  │
│                              ▼                                               │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                     WebSocket Handler (Port 8765)                      │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │  Session Manager                                                 │  │  │
│  │  │  - Creates isolated Session per connection                       │  │  │
│  │  │  - Each session has: audio_buffer, speaker_identifier, timing    │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                              │                                               │
│                              ▼                                               │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      Processing Pipeline                               │  │
│  │                                                                        │  │
│  │   Audio Chunks ──► AudioBuffer ──► Transcription ──► Speaker ID       │  │
│  │   (16-bit PCM)     (3 sec)        (Faster Whisper)   (Pyannote)       │  │
│  │                                                                        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                              │                                               │
│                              ▼                                               │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         ML Models (GPU)                                │  │
│  │                                                                        │  │
│  │   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐   │  │
│  │   │  Whisper Model  │    │   Silero VAD    │    │ Pyannote Embed  │   │  │
│  │   │  (large-v3)     │    │  (built-in)     │    │ (speaker ID)    │   │  │
│  │   └─────────────────┘    └─────────────────┘    └─────────────────┘   │  │
│  │                                                                        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      HTTP API (Port 8080)                              │  │
│  │   GET  /health     - Health check + active session count              │  │
│  │   POST /transcribe - Batch transcription for file uploads             │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Docker with NVIDIA GPU support
- NVIDIA GPU with CUDA support
- [HuggingFace token](https://huggingface.co/settings/tokens) (for speaker identification)

### 1. Clone and Configure

```bash
git clone <repository>
cd transcribe

# Copy example environment file
cp .env.example .env

# Edit .env and add your HuggingFace token
nano .env
```

### 2. Accept Model Terms

Before building, accept the terms for the speaker embedding model:
- https://huggingface.co/pyannote/embedding

### 3. Build and Run

```bash
# Build (downloads models during build - takes a few minutes first time)
docker-compose build

# Start the server
docker-compose up -d

# Check logs
docker-compose logs -f
```

### 4. Verify It's Running

```bash
curl http://localhost:8080/health
```

Expected response:
```json
{
  "status": "ok",
  "model": "large-v3",
  "speaker_identification_enabled": true,
  "speaker_identification_loaded": true,
  "active_sessions": 0
}
```

---

## 📡 API Reference

### WebSocket Streaming API

**Endpoint:** `ws://localhost:8765`

#### Connection Flow

```
Client                                Server
  │                                     │
  │──── Connect ────────────────────────►│
  │                                     │
  │◄─── session_start ──────────────────│
  │     {                               │
  │       "type": "session_start",      │
  │       "session_id": "uuid...",      │
  │       "speaker_identification": true │
  │     }                               │
  │                                     │
  │──── Audio chunks (binary) ──────────►│
  │──── Audio chunks (binary) ──────────►│
  │──── Audio chunks (binary) ──────────►│
  │                                     │
  │◄─── Transcript ─────────────────────│
  │     {                               │
  │       "session_id": "uuid...",      │
  │       "transcript": "Hello world",  │
  │       "start": 0.5,                 │
  │       "end": 1.2,                   │
  │       "speaker": "SPEAKER_00",      │
  │       "is_final": true              │
  │     }                               │
  │                                     │
  │──── Close ──────────────────────────►│
```

#### Audio Format Requirements

| Property | Value |
|----------|-------|
| Sample Rate | 16000 Hz |
| Bit Depth | 16-bit signed integer |
| Channels | Mono |
| Encoding | Raw PCM (no headers) |

#### Example: Python WebSocket Client

```python
import asyncio
import websockets
import numpy as np

async def transcribe_audio():
    async with websockets.connect("ws://localhost:8765") as ws:
        # Receive session info
        session_info = await ws.recv()
        print(f"Connected: {session_info}")
        
        # Send audio chunks (example: from microphone or file)
        # Audio must be 16-bit PCM, 16kHz, mono
        audio_chunk = np.zeros(16000, dtype=np.int16)  # 1 second of silence
        await ws.send(audio_chunk.tobytes())
        
        # Receive transcripts
        async for message in ws:
            print(f"Transcript: {message}")

asyncio.run(transcribe_audio())
```

#### Example: JavaScript WebSocket Client

```javascript
const ws = new WebSocket('ws://localhost:8765');

ws.onopen = () => console.log('Connected');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'session_start') {
    console.log('Session started:', data.session_id);
  } else {
    console.log(`[${data.speaker}] ${data.transcript}`);
  }
};

// Send audio from MediaRecorder (must be converted to 16-bit PCM)
// See examples/ folder for complete implementation
```

---

### HTTP Transcription API

**Endpoint:** `POST /transcribe`

Upload an audio file for batch transcription.

#### Request

```bash
curl -X POST http://localhost:8080/transcribe \
  -H "Content-Type: audio/wav" \
  --data-binary @audio.raw
```

**Note:** Audio must be raw 16-bit PCM, 16kHz, mono (no WAV header).

#### Response

```json
{
  "segments": [
    {
      "text": "Hello, how are you?",
      "start": 0.5,
      "end": 1.8,
      "speaker": "SPEAKER_00"
    },
    {
      "text": "I'm doing great, thanks!",
      "start": 2.1,
      "end": 3.5,
      "speaker": "SPEAKER_01"
    }
  ],
  "total_segments": 2,
  "full_text": "Hello, how are you? I'm doing great, thanks!",
  "has_speakers": true
}
```

---

### Health Check API

**Endpoint:** `GET /health`

#### Response

```json
{
  "status": "ok",
  "model": "large-v3",
  "speaker_identification_enabled": true,
  "speaker_identification_loaded": true,
  "active_sessions": 3
}
```

---

## 🔧 Processing Pipeline

### Audio Processing Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AUDIO PROCESSING PIPELINE                          │
└─────────────────────────────────────────────────────────────────────────────┘

1. AUDIO INPUT
   ┌─────────────────────────────────────────────────────────────────────────┐
   │  Raw Audio Chunks (16-bit PCM, 16kHz, mono)                             │
   │  - Arrives via WebSocket as binary messages                             │
   │  - Typical chunk size: 100ms - 500ms                                    │
   └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
2. AUDIO BUFFERING
   ┌─────────────────────────────────────────────────────────────────────────┐
   │  AudioBuffer (per-session)                                              │
   │  - Efficient deque-based buffer                                         │
   │  - Accumulates chunks until CHUNK_SIZE_SECONDS (default: 3.0s)          │
   │  - Prevents memory overflow with MAX_SEGMENT_SECONDS limit              │
   └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
3. NORMALIZATION
   ┌─────────────────────────────────────────────────────────────────────────┐
   │  Audio Normalization                                                    │
   │  - Convert int16 → float32                                              │
   │  - Normalize amplitude to 0.95 peak                                     │
   │  - Boosts soft voices for better recognition                            │
   └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
4. VOICE ACTIVITY DETECTION (VAD)
   ┌─────────────────────────────────────────────────────────────────────────┐
   │  Silero VAD (built into faster-whisper)                                 │
   │  - Detects speech vs silence                                            │
   │  - Parameters:                                                          │
   │    • min_silence_duration_ms: 500 (end of utterance detection)          │
   │    • speech_pad_ms: 200 (padding around speech)                         │
   └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
5. TRANSCRIPTION
   ┌─────────────────────────────────────────────────────────────────────────┐
   │  Faster Whisper (GPU accelerated)                                       │
   │  - Model: large-v3 (configurable)                                       │
   │  - Beam search decoding (beam_size: 5)                                  │
   │  - Auto language detection or fixed language                            │
   │  - Word-level timestamps enabled                                        │
   │  - Hallucination filtering (repetition detection)                       │
   └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
6. SPEAKER IDENTIFICATION
   ┌─────────────────────────────────────────────────────────────────────────┐
   │  Pyannote Embedding Model                                               │
   │  - Extract voice embedding for each speech segment                      │
   │  - Compare with known speaker voiceprints                               │
   │  - Auto-enroll new speakers (SPEAKER_00, SPEAKER_01, ...)               │
   │  - Adaptive voiceprint updates with exponential moving average          │
   └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
7. OUTPUT
   ┌─────────────────────────────────────────────────────────────────────────┐
   │  JSON Response via WebSocket                                            │
   │  {                                                                      │
   │    "session_id": "uuid",                                                │
   │    "transcript": "transcribed text",                                    │
   │    "start": 0.0,                                                        │
   │    "end": 1.5,                                                          │
   │    "speaker": "SPEAKER_00",                                             │
   │    "is_final": true                                                     │
   │  }                                                                      │
   └─────────────────────────────────────────────────────────────────────────┘
```

---

## 🎭 Speaker Identification

### How It Works

The speaker identification system uses **voice embeddings** to identify who is speaking:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SPEAKER IDENTIFICATION FLOW                           │
└─────────────────────────────────────────────────────────────────────────────┘

1. EMBEDDING EXTRACTION
   ┌─────────────────────────────────────────────────────────────────────────┐
   │  For each transcribed segment:                                          │
   │  - Extract audio for that segment                                       │
   │  - Pass through Pyannote embedding model                                │
   │  - Get 512-dimensional voice embedding (normalized)                     │
   └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
2. SPEAKER MATCHING
   ┌─────────────────────────────────────────────────────────────────────────┐
   │  Compare embedding with known speakers:                                 │
   │  - Calculate cosine similarity with each speaker's voiceprint           │
   │  - If similarity >= SIMILARITY_THRESHOLD (0.70): Match found            │
   │  - If similarity < ENROLLMENT_THRESHOLD (0.65): New speaker             │
   │  - Otherwise: Uncertain, skip assignment                                │
   └─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
3a. EXISTING SPEAKER                    3b. NEW SPEAKER
   ┌────────────────────────┐              ┌────────────────────────┐
   │  Update voiceprint     │              │  Create pending speaker │
   │  with EMA:             │              │  (SPEAKER_XX)           │
   │  new = (1-α)·old + α·e │              │                        │
   │                        │              │  Requires N matches     │
   │  Return speaker label  │              │  before confirmation    │
   └────────────────────────┘              └────────────────────────┘
```

### Speaker Lifecycle

```
PENDING                          CONFIRMED
┌─────────┐                     ┌─────────┐
│ New     │  N consistent       │Confirmed│
│ Speaker │ ────matches────────►│ Speaker │
└─────────┘                     └─────────┘
     │                               │
     │ Not enough matches            │ Voiceprint updates
     │ or noise                      │ with each match
     ▼                               ▼
┌─────────┐                     ┌─────────┐
│Discarded│                     │ Stable  │
│         │                     │ Speaker │
└─────────┘                     └─────────┘
```

### Tunable Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SPEAKER_SIMILARITY_THRESHOLD` | 0.70 | Cosine similarity for matching |
| `SPEAKER_ENROLLMENT_THRESHOLD` | 0.65 | Below this, enroll new speaker |
| `SPEAKER_CONFIRMATION_COUNT` | 2 | Matches before confirming |
| `SPEAKER_MIN_SEGMENT_DURATION` | 0.5s | Minimum audio for embedding |
| `SPEAKER_VOICEPRINT_MEMORY` | 20 | Embeddings to remember |
| `SPEAKER_LEARNING_RATE` | 0.15 | Voiceprint adaptation speed |

---

## 🏢 Multi-Tenancy

The server supports multiple concurrent transcription sessions with complete isolation:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            MULTI-TENANT ARCHITECTURE                         │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────────┐
                    │        Shared Resources             │
                    │  - Whisper Model (stateless)        │
                    │  - Pyannote Embedding Model         │
                    │  - Thread Pool Executor             │
                    └─────────────────────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          │                         │                         │
          ▼                         ▼                         ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│     Session A       │  │     Session B       │  │     Session C       │
│  ─────────────────  │  │  ─────────────────  │  │  ─────────────────  │
│  session_id: uuid-a │  │  session_id: uuid-b │  │  session_id: uuid-c │
│  audio_buffer: [...] │  │  audio_buffer: [...] │  │  audio_buffer: [...] │
│  audio_queue: Queue │  │  audio_queue: Queue │  │  audio_queue: Queue │
│  speaker_id: {...}  │  │  speaker_id: {...}  │  │  speaker_id: {...}  │
│  time_offset: 45.2s │  │  time_offset: 12.8s │  │  time_offset: 3.1s  │
│                     │  │                     │  │                     │
│  Speakers:          │  │  Speakers:          │  │  Speakers:          │
│  - SPEAKER_00 (John)│  │  - SPEAKER_00 (Lisa)│  │  - SPEAKER_00       │
│  - SPEAKER_01 (Jane)│  │  - SPEAKER_01 (Mike)│  │                     │
└─────────────────────┘  └─────────────────────┘  └─────────────────────┘
```

Each session is completely independent:
- **Audio buffers** don't mix between sessions
- **Speaker IDs** are scoped to each session (SPEAKER_00 in Session A ≠ SPEAKER_00 in Session B)
- **Timestamps** are relative to each session's start
- **Logging** includes session ID prefix for debugging

---

## 📁 Project Structure

```
transcribe/
├── server.py                    # Main entry point
├── Dockerfile                   # Container definition
├── docker-compose.yml           # Orchestration config
├── requirements.txt             # Python dependencies
├── .env.example                 # Example environment config
│
├── app/
│   ├── config/
│   │   └── settings.py          # Configuration from env vars
│   │
│   ├── handlers/
│   │   ├── websocket_handler.py # WebSocket connection handler
│   │   ├── transcription_api.py # HTTP POST /transcribe
│   │   └── health_check.py      # HTTP GET /health
│   │
│   ├── services/
│   │   ├── transcription.py     # Faster Whisper wrapper
│   │   ├── speaker_identification.py  # Voice embedding + speaker ID
│   │   └── text_correction.py   # (Optional) Text post-processing
│   │
│   └── utils/
│       ├── logger.py            # Logging configuration
│       ├── hf_hub_compat.py     # HuggingFace Hub compatibility
│       └── torchaudio_compat.py # Torchaudio compatibility
│
└── scripts/
    └── download_models.py       # Pre-download models during build
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| **Server** | | |
| `SERVER_HOST` | `0.0.0.0` | Bind address |
| `HEALTH_CHECK_PORT` | `8080` | HTTP API port |
| `WEBSOCKET_PORT` | `8765` | WebSocket port |
| **Model** | | |
| `MODEL_NAME` | `large-v3` | Whisper model size |
| `MODEL_PATH` | `/app/models` | Model cache directory |
| `LANGUAGE` | (auto) | Force language (e.g., `en`, `nl`) |
| **Device** | | |
| `DEVICE` | `cuda` | `cuda` or `cpu` |
| `COMPUTE_TYPE` | `float16` | `float16`, `int8`, `float32` |
| `PROCESSING_THREADS` | `4` | Thread pool size |
| **VAD** | | |
| `VAD_MIN_SILENCE_MS` | `500` | Silence to end utterance |
| `VAD_SPEECH_PAD_MS` | `200` | Padding around speech |
| **Transcription** | | |
| `BEAM_SIZE` | `5` | Beam search width |
| `BEST_OF` | `1` | Candidates to consider |
| **Speaker ID** | | |
| `USE_DIARIZATION` | `true` | Enable speaker identification |
| `HF_TOKEN` | | HuggingFace token (required) |
| `SPEAKER_SIMILARITY_THRESHOLD` | `0.70` | Match threshold |
| `SPEAKER_ENROLLMENT_THRESHOLD` | `0.65` | New speaker threshold |
| `SPEAKER_CONFIRMATION_COUNT` | `2` | Matches to confirm |
| `SPEAKER_MIN_SEGMENT_DURATION` | `0.5` | Min segment (seconds) |
| `SPEAKER_VOICEPRINT_MEMORY` | `20` | Embeddings per speaker |
| `SPEAKER_LEARNING_RATE` | `0.15` | Voiceprint adaptation |

---

## 🐳 Docker Details

### Build Arguments

```bash
# Build with specific model
MODEL_NAME=medium docker-compose build

# Build with speaker identification
HF_TOKEN=your_token docker-compose build
```

### Resource Allocation

The `docker-compose.yml` is optimized for:
- **GPU:** NVIDIA RTX 3070 (8GB VRAM)
- **CPU:** Intel i9-9900K (8 cores, 16 threads)
- **RAM:** 32GB system memory

Adjust `deploy.resources` section for your hardware.

### Volume Mounts

| Volume | Purpose |
|--------|---------|
| `~/.cache/huggingface` | Share HF cache with host |
| `~/.cache/torch` | Share PyTorch cache |
| `transcription-models` | Persistent model storage |

---

## 📊 Performance Tuning

### Latency vs Accuracy Trade-offs

| Setting | Lower Latency | Higher Accuracy |
|---------|---------------|-----------------|
| `CHUNK_SIZE_SECONDS` | 1.5 - 2.0 | 3.0 - 5.0 |
| `BEAM_SIZE` | 1 - 3 | 5 - 10 |
| `BEST_OF` | 1 | 3 - 5 |
| `MODEL_NAME` | tiny, base | large-v3 |

### Memory Usage

| Model | VRAM Required | RAM Required |
|-------|---------------|--------------|
| tiny | ~1 GB | ~2 GB |
| base | ~1.5 GB | ~3 GB |
| small | ~2 GB | ~4 GB |
| medium | ~5 GB | ~8 GB |
| large-v3 | ~6 GB | ~10 GB |

---

## 🔍 Troubleshooting

### Common Issues

**Model not loading:**
```bash
# Check GPU availability
docker-compose exec transcription-server nvidia-smi

# Check logs
docker-compose logs -f
```

**Speaker identification not working:**
```bash
# Verify HF_TOKEN is set
docker-compose exec transcription-server env | grep HF_TOKEN

# Check if model loaded
curl http://localhost:8080/health | jq .speaker_identification_loaded
```

**High latency:**
- Reduce `CHUNK_SIZE_SECONDS` for faster response
- Reduce `BEAM_SIZE` for faster decoding
- Use smaller model (`medium` instead of `large-v3`)

**Memory errors:**
- Increase Docker's `shm_size`
- Use smaller model
- Reduce `SPEAKER_VOICEPRINT_MEMORY`

---

## 📜 License

[Add your license here]

---

## 🙏 Acknowledgments

- [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) - CTranslate2-based Whisper implementation
- [Pyannote Audio](https://github.com/pyannote/pyannote-audio) - Speaker diarization toolkit
- [OpenAI Whisper](https://github.com/openai/whisper) - Original Whisper model

