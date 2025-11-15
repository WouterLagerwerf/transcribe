"""Configuration settings loaded from environment variables."""

import os

# Server Configuration
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
HEALTH_CHECK_PORT = int(os.getenv("HEALTH_CHECK_PORT", 8080))

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

# faster-whisper configuration
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")  # "int8", "float16", "float32"

# Speaker Diarization Configuration
USE_DIARIZATION = os.getenv("USE_DIARIZATION", "true").lower() == "true"
HF_TOKEN = os.getenv("HF_TOKEN", None)  # HuggingFace token for private models (optional)

