"""Configuration settings loaded from environment variables."""

import os
import torch
import sys

# Server Configuration
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
HEALTH_CHECK_PORT = int(os.getenv("HEALTH_CHECK_PORT", 8080))

# Device Configuration (GPU/CPU)
# Auto-detect CUDA availability, but allow override via environment variable
_device_override = os.getenv("DEVICE", None)
if _device_override:
    DEVICE = _device_override.lower()  # "cuda" or "cpu"
    _device_source = "environment variable override"
else:
    # Auto-detect: use CUDA if available, otherwise CPU
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    _device_source = "auto-detection"

# Prominent logging of device configuration at import time
print("=" * 80, file=sys.stderr)
print("🚀 TRANSCRIPTION SERVER - DEVICE CONFIGURATION", file=sys.stderr)
print("=" * 80, file=sys.stderr)
print(f"📱 Device: {DEVICE.upper()}", file=sys.stderr)
print(f"   Source: {_device_source}", file=sys.stderr)

if DEVICE == "cuda":
    if torch.cuda.is_available():
        print(f"   ✅ CUDA Available: YES", file=sys.stderr)
        print(f"   🎮 GPU Device: {torch.cuda.get_device_name(0)}", file=sys.stderr)
        print(f"   💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB", file=sys.stderr)
        print(f"   🔢 CUDA Version: {torch.version.cuda}", file=sys.stderr)
    else:
        print(f"   ⚠️  CUDA requested but not available - falling back to CPU", file=sys.stderr)
        DEVICE = "cpu"
else:
    print(f"   ℹ️  Using CPU for inference", file=sys.stderr)
    if torch.cuda.is_available():
        print(f"   💡 Note: CUDA is available but not being used (set DEVICE=cuda to enable)", file=sys.stderr)

print("=" * 80, file=sys.stderr)

# Model Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "base")
MODEL_PATH = os.getenv("MODEL_PATH", ".")
# LANGUAGE: None for auto-detection, empty string is converted to None
_language = os.getenv("LANGUAGE", None)
LANGUAGE = None if _language == "" or _language is None else _language
PROCESSING_THREADS = int(os.getenv("PROCESSING_THREADS", 4))

# Audio Processing Configuration
SAMPLE_RATE = 16000
CHUNK_SIZE_SECONDS = 3.0  # Duration of audio segment to process at once (fallback if VAD disabled)
MAX_SEGMENT_SECONDS = float(os.getenv("MAX_SEGMENT_SECONDS", "3600.0"))  # Maximum segment size (60 minutes default)

# VAD Configuration (Voice Activity Detection)
USE_VAD = os.getenv("USE_VAD", "true").lower() == "true"
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))
VAD_MIN_SILENCE_MS = int(os.getenv("VAD_MIN_SILENCE_MS", "500"))  # Pauses to detect end of utterance
VAD_SPEECH_PAD_MS = int(os.getenv("VAD_SPEECH_PAD_MS", "100"))  # Padding around speech
VAD_MAX_SPEECH_MS = int(os.getenv("VAD_MAX_SPEECH_MS", "5000"))  # Max speech duration before forcing transcription
# Real-time transcription: transcribe every N seconds during continuous speech (for live updates)
REALTIME_INTERVAL_SECONDS = float(os.getenv("REALTIME_INTERVAL_SECONDS", "0.8"))  # Transcribe every 0.8 seconds during speech for faster partials

# faster-whisper configuration
# Auto-select compute type based on device: float16 for GPU, int8 for CPU (unless overridden)
_compute_type_override = os.getenv("COMPUTE_TYPE", None)
if _compute_type_override:
    COMPUTE_TYPE = _compute_type_override  # "int8", "float16", "float32", "int8_float16"
else:
    # Default: float16 for GPU (best performance), int8 for CPU (best performance)
    COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

# Speaker Diarization Configuration
USE_DIARIZATION = os.getenv("USE_DIARIZATION", "true").lower() == "true"
HF_TOKEN = os.getenv("HF_TOKEN", None)  # HuggingFace token for private models (optional)

# Speaker Enrollment Configuration
ENROLLMENT_SIMILARITY_THRESHOLD = float(os.getenv("ENROLLMENT_SIMILARITY_THRESHOLD", "0.72"))  # Cosine similarity threshold (0.0-1.0, higher = stricter matching)
ENROLLMENT_MIN_SEGMENT_DURATION = float(os.getenv("ENROLLMENT_MIN_SEGMENT_DURATION", "0.35"))  # Minimum segment duration in seconds for reliable embedding
ENROLLMENT_LEARNING_RATE = float(os.getenv("ENROLLMENT_LEARNING_RATE", "0.2"))  # Learning rate for voiceprint updates (0.0-1.0, lower = more stable, higher = adapts faster)
ENROLLMENT_MIN_CONFIDENCE = float(os.getenv("ENROLLMENT_MIN_CONFIDENCE", "0.68"))  # Minimum confidence to enroll new speaker (prevents noise enrollment)
ENROLLMENT_ADAPTIVE_THRESHOLD = os.getenv("ENROLLMENT_ADAPTIVE_THRESHOLD", "true").lower() == "true"  # Adjust threshold based on number of speakers
ENROLLMENT_MAX_SPEAKERS = int(os.getenv("ENROLLMENT_MAX_SPEAKERS", "10"))  # Maximum number of speakers to enroll
ENROLLMENT_MIN_SAMPLES = int(os.getenv("ENROLLMENT_MIN_SAMPLES", "2"))  # Minimum number of embeddings before considering speaker "stable"
ENROLLMENT_CONFIDENCE_WINDOW = int(os.getenv("ENROLLMENT_CONFIDENCE_WINDOW", "5"))  # Number of recent similarities to track for confidence

