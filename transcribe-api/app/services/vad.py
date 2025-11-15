"""Voice Activity Detection (VAD) using Silero VAD model."""

import numpy as np
import torch
from app.config.settings import SAMPLE_RATE, VAD_THRESHOLD
from app.utils.logger import logger

# Global state
vad_model = None
vad_loaded = False

# Silero VAD requires exactly 512 samples for 16kHz (or 256 for 8kHz)
VAD_WINDOW_SAMPLES = 512 if SAMPLE_RATE == 16000 else 256


def load_vad_model():
    """Loads the Silero VAD model into memory."""
    global vad_model, vad_loaded
    if vad_loaded:
        return
    
    logger.info("Loading Silero VAD model...")
    try:
        vad_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        vad_model.eval()  # Set to evaluation mode for inference
        vad_loaded = True
        logger.info("Silero VAD model loaded successfully.")
    except Exception as e:
        logger.warning(f"Failed to load Silero VAD model: {e}. Server will use time-based transcription.")
        logger.warning("VAD functionality will be disabled. Check that torchaudio is installed.")
        vad_loaded = False
        # Don't raise - allow server to continue without VAD


def detect_speech(audio_float32: np.ndarray) -> float:
    """
    Detects speech in audio chunk.
    
    Args:
        audio_float32: Audio data as float32 numpy array, normalized to [-1, 1]
                      Must be exactly VAD_WINDOW_SAMPLES (512 for 16kHz, 256 for 8kHz)
    
    Returns:
        Speech probability (0.0 to 1.0)
    """
    if vad_model is None:
        raise RuntimeError("VAD model not loaded. Call load_vad_model() first.")
    
    # Ensure we have exactly the right number of samples
    if len(audio_float32) != VAD_WINDOW_SAMPLES:
        # Pad or truncate to required size
        if len(audio_float32) < VAD_WINDOW_SAMPLES:
            # Pad with zeros
            padded = np.zeros(VAD_WINDOW_SAMPLES, dtype=np.float32)
            padded[:len(audio_float32)] = audio_float32
            audio_float32 = padded
        else:
            # Take the last VAD_WINDOW_SAMPLES
            audio_float32 = audio_float32[-VAD_WINDOW_SAMPLES:]
    
    try:
        with torch.no_grad():
            # Convert numpy array to torch tensor
            audio_tensor = torch.from_numpy(audio_float32)
            # Run VAD model
            speech_prob = vad_model(audio_tensor, SAMPLE_RATE).item()
        return speech_prob
    except Exception as e:
        logger.error(f"Error during VAD detection: {e}", exc_info=True)
        return 0.0


def detect_speech_in_chunk(audio_int16: np.ndarray) -> float:
    """
    Detects speech in an audio chunk of any size.
    Extracts the last VAD_WINDOW_SAMPLES for detection.
    
    Args:
        audio_int16: Audio data as int16 numpy array
    
    Returns:
        Speech probability (0.0 to 1.0)
    """
    # Convert to float32 and normalize
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    return detect_speech(audio_float32)


def is_speech(audio_float32: np.ndarray) -> bool:
    """
    Determines if audio contains speech based on threshold.
    
    Args:
        audio_float32: Audio data as float32 numpy array, normalized to [-1, 1]
    
    Returns:
        True if speech detected, False otherwise
    """
    speech_prob = detect_speech(audio_float32)
    return speech_prob > VAD_THRESHOLD


def is_vad_loaded() -> bool:
    """Returns whether VAD model is loaded."""
    return vad_loaded

