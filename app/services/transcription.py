"""faster-whisper model loading and transcription logic."""

import numpy as np
from faster_whisper import WhisperModel
from concurrent.futures import ThreadPoolExecutor

from app.config.settings import (
    MODEL_NAME, MODEL_PATH, LANGUAGE, PROCESSING_THREADS, COMPUTE_TYPE, DEVICE
)
from app.utils.logger import logger

# Global state
whisper_model = None
executor = ThreadPoolExecutor(max_workers=PROCESSING_THREADS)
server_ready = False


def load_model():
    """Loads the faster-whisper model into memory. This is a blocking operation."""
    global whisper_model, server_ready
    import sys
    
    print("", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print(f"🎤 LOADING WHISPER MODEL: {MODEL_NAME}", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print(f"   Device: {DEVICE.upper()}", file=sys.stderr)
    print(f"   Compute Type: {COMPUTE_TYPE}", file=sys.stderr)
    if DEVICE == "cpu":
        print(f"   CPU Threads: {PROCESSING_THREADS}", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    
    logger.info(f"Loading faster-whisper model '{MODEL_NAME}' on device '{DEVICE}' with compute_type '{COMPUTE_TYPE}'...")
    try:
        # Build kwargs - only include cpu_threads for CPU device
        model_kwargs = {
            "device": DEVICE,
            "compute_type": COMPUTE_TYPE,
            "download_root": MODEL_PATH
        }
        if DEVICE == "cpu":
            model_kwargs["cpu_threads"] = PROCESSING_THREADS
        
        whisper_model = WhisperModel(MODEL_NAME, **model_kwargs)
        print("", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print(f"✅ WHISPER MODEL LOADED SUCCESSFULLY", file=sys.stderr)
        print(f"   Model: {MODEL_NAME}", file=sys.stderr)
        print(f"   Device: {DEVICE.upper()}", file=sys.stderr)
        print(f"   Compute Type: {COMPUTE_TYPE}", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print("", file=sys.stderr)
        logger.info(f"faster-whisper model '{MODEL_NAME}' loaded successfully on {DEVICE}.")
        server_ready = True
    except Exception as e:
        print("", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print(f"❌ FAILED TO LOAD WHISPER MODEL", file=sys.stderr)
        print(f"   Error: {e}", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print("", file=sys.stderr)
        logger.error(f"Failed to load faster-whisper model: {e}", exc_info=True)
        # The server will not be marked as ready and health checks will fail.
        # You might want to exit here in a containerized environment.
        # exit(1)


def transcribe_synchronous(audio_data_float32: np.ndarray, time_offset: float = 0.0):
    """
    Synchronous wrapper for faster-whisper transcription.
    This function will be run in a separate thread by the executor.
    faster-whisper is significantly faster and more accurate than standard Whisper.
    
    Args:
        audio_data_float32: Audio data as float32 numpy array
        time_offset: Time offset in seconds to add to segment timestamps (for absolute timing)
    
    Returns:
        List of dicts with 'text', 'start', 'end' keys, or empty list on error
    """
    if whisper_model is None:
        logger.error("Transcription called but model is not loaded.")
        return []
    try:
        # faster-whisper returns segments generator
        segments, info = whisper_model.transcribe(
            audio_data_float32,
            beam_size=5,  # Higher beam size for better accuracy
            language=LANGUAGE if LANGUAGE else None,
            vad_filter=False,  # We handle VAD separately
            vad_parameters=None
        )
        # Extract segments with timestamps
        result = []
        for segment in segments:
            result.append({
                "text": segment.text.strip(),
                "start": segment.start + time_offset,
                "end": segment.end + time_offset
            })
        logger.debug(f"Transcription completed: language={info.language}, probability={info.language_probability:.2f}, segments={len(result)}")
        return result
    except Exception as e:
        logger.error(f"Error during transcription: {e}", exc_info=True)
        return []


def get_executor():
    """Returns the thread pool executor."""
    return executor


def is_server_ready():
    """Returns whether the server is ready (model loaded)."""
    return server_ready


def get_model_name():
    """Returns the current model name."""
    return MODEL_NAME

