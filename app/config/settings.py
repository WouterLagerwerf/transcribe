"""Configuration settings loaded from environment variables."""

import os
import torch
import sys

# VAD selection (pyannote or silero)
VAD_METHOD = os.getenv("VAD_METHOD", "pyannote")
VAD_ONSET = float(os.getenv("VAD_ONSET", "0.50"))
VAD_OFFSET = float(os.getenv("VAD_OFFSET", "0.36"))
VAD_CHUNK_SIZE = int(os.getenv("VAD_CHUNK_SIZE", "30"))

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

# Cached torch device object (avoid repeated torch.device() calls)
TORCH_DEVICE = torch.device(DEVICE)

# Model Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "base")
MODEL_PATH = os.getenv("MODEL_PATH", ".")
# LANGUAGE: None for auto-detection, empty string is converted to None
_language = os.getenv("LANGUAGE", None)
LANGUAGE = None if _language == "" or _language is None else _language
PROCESSING_THREADS = int(os.getenv("PROCESSING_THREADS", 4))

# Audio Processing Configuration
SAMPLE_RATE = 16000
CHUNK_SIZE_SECONDS = 3.0  # Duration of audio segment to process at once
MAX_SEGMENT_SECONDS = float(os.getenv("MAX_SEGMENT_SECONDS", "3600.0"))  # Maximum segment size (60 minutes default)

# VAD Configuration (Voice Activity Detection)
# Uses faster-whisper's built-in VAD for speech detection
VAD_MIN_SILENCE_MS = int(os.getenv("VAD_MIN_SILENCE_MS", "500"))  # Minimum silence to detect end of speech
VAD_SPEECH_PAD_MS = int(os.getenv("VAD_SPEECH_PAD_MS", "200"))  # Padding around speech segments

# faster-whisper configuration
# Auto-select compute type based on device (unless overridden)
_compute_type_override = os.getenv("COMPUTE_TYPE", None)
if _compute_type_override:
    COMPUTE_TYPE = _compute_type_override  # "int8", "float16", "float32", "int8_float16"
else:
    # Default: float16 for both GPU and CPU
    COMPUTE_TYPE = "float16"

# Transcription parameters (tunable for latency vs accuracy trade-off)
BEAM_SIZE = int(os.getenv("BEAM_SIZE", "5"))  # Beam size for decoding (higher = more accurate, slower)
BEST_OF = int(os.getenv("BEST_OF", "1"))  # Number of candidates to consider (higher = more accurate, slower)

# Speaker Identification Configuration
# When enabled, uses pyannote embedding model to identify speakers from their voice
USE_DIARIZATION = os.getenv("USE_DIARIZATION", "true").lower() == "true"
HF_TOKEN = os.getenv("HF_TOKEN", None)  # HuggingFace token for private models (required for speaker ID)

# Diarization model (overlap-capable)
DIARIZATION_MODEL = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")

# Alignment configuration
USE_ALIGNMENT = os.getenv("USE_ALIGNMENT", "false").lower() == "true"
ALIGN_MODEL_NAME = os.getenv("ALIGN_MODEL", None)

# Note: Speaker identification tuning parameters (SPEAKER_*) are loaded directly in speaker_identification.py
# This avoids circular imports and keeps configuration close to the code that uses it
