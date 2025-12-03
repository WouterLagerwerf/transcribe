# -*- coding: utf-8 -*-
"""faster-whisper model loading and transcription logic."""

import numpy as np
from faster_whisper import WhisperModel
from concurrent.futures import ThreadPoolExecutor

from app.config.settings import (
    MODEL_NAME, MODEL_PATH, LANGUAGE, PROCESSING_THREADS, COMPUTE_TYPE, DEVICE,
    BEAM_SIZE, BEST_OF, VAD_MIN_SILENCE_MS, VAD_SPEECH_PAD_MS
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
        # Normalize audio to improve detection of soft/unclear voices
        # This amplifies quiet audio while preventing clipping
        # Optimized: Work directly in float32 to avoid costly type conversions
        audio_max = np.max(np.abs(audio_data_float32))
        if audio_max > 1e-6:  # Avoid division by very small numbers
            # Normalize to 0.95 to leave headroom and boost soft voices
            # In-place multiplication is faster than creating intermediate arrays
            audio_normalized = audio_data_float32 * (0.95 / audio_max)
        else:
            audio_normalized = audio_data_float32
        
        # faster-whisper returns segments generator
        # Parameters balanced to reject uncertain/weird predictions while allowing legitimate speech
        # Use auto language detection if LANGUAGE is not set
        detected_language = LANGUAGE if LANGUAGE else None
        
        # Built-in VAD parameters (uses faster-whisper's Silero VAD internally)
        vad_params = {
            "min_silence_duration_ms": VAD_MIN_SILENCE_MS,
            "speech_pad_ms": VAD_SPEECH_PAD_MS,
        }
        
        segments, info = whisper_model.transcribe(
            audio_normalized,
            beam_size=BEAM_SIZE,  # Configurable beam size (default 5, higher = more accurate but slower)
            best_of=BEST_OF,  # Configurable candidates (default 1, higher = more accurate but slower)
            language=detected_language,  # None = auto-detect, otherwise use specified language
            vad_filter=True,  # Always use faster-whisper's built-in VAD
            vad_parameters=vad_params,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),  # Temperature fallback for accuracy
            condition_on_previous_text=True,  # Use context from previous text for better articulation
            no_speech_threshold=0.4,  # Balanced threshold to filter uncertain segments
            log_prob_threshold=-0.7,  # Balanced threshold to reject uncertain predictions
            compression_ratio_threshold=2.4,  # Detect repetitive/hallucinated text
            suppress_blank=True,  # Suppress blank outputs
            initial_prompt=None,  # Don't bias towards any specific phrases
            word_timestamps=True,  # Enable word-level timestamps for better accuracy
        )
        
        # Log detected language for debugging
        if detected_language is None and hasattr(info, 'language'):
            logger.debug(f"Auto-detected language: {info.language} (probability: {getattr(info, 'language_probability', 'N/A')})")
        # Extract segments with timestamps
        # Filter out very short segments and low-confidence segments that are likely noise
        result = []
        for segment in segments:
            text = segment.text.strip()
            
            # Skip empty segments
            if not text:
                continue
            
            # Skip very short segments (< 0.25 seconds) that are likely noise or uncertain predictions
            segment_duration = segment.end - segment.start
            if segment_duration < 0.25:
                logger.debug(f"Filtered short segment (likely noise/uncertain): '{text}' ({segment_duration:.2f}s)")
                continue
            
            # Skip segments with low average log probability (likely noise/hallucination/uncertain predictions)
            # Balanced threshold to reject uncertain/weird outputs while allowing legitimate speech
            if hasattr(segment, 'avg_logprob') and segment.avg_logprob < -1.0:
                logger.debug(f"Filtered low-confidence segment (uncertain/weird): '{text}' (logprob: {segment.avg_logprob:.2f})")
                continue
            
            # Filter out repetitive/hallucinated text (e.g., "I'm sorry" repeated many times)
            # Check if text contains the same phrase repeated many times
            words = text.split()
            if len(words) >= 6:  # Need at least 6 words to detect repetition
                # Check for repetitive phrases - look for patterns where the same 2-word phrase repeats
                # This catches cases like "I'm sorry, I'm sorry, I'm sorry..."
                first_phrase = ' '.join(words[:2]).lower()
                phrase_count = 1
                for i in range(2, len(words) - 1, 2):
                    phrase = ' '.join(words[i:i+2]).lower()
                    if phrase == first_phrase:
                        phrase_count += 1
                    else:
                        break
                # If the same 2-word phrase appears 3+ times consecutively, it's likely a hallucination
                if phrase_count >= 3:
                    logger.debug(f"Filtered repetitive segment (hallucination): '{text[:80]}...' (phrase '{first_phrase}' repeated {phrase_count} times)")
                    continue
            
            result.append({
                "text": text,
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

