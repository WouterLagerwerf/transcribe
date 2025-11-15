"""Speaker diarization using pyannote.audio."""

# IMPORTANT: Import compatibility shims FIRST before any other imports
# These patches are required for pyannote.audio to work with newer library versions

# 1. Patch huggingface_hub to accept use_auth_token (must be before pyannote imports hf_hub)
from app.utils.hf_hub_compat import *  # noqa: F403, F401

# 2. Patch torchaudio.AudioMetaData which pyannote.audio requires
from app.utils.torchaudio_compat import *  # noqa: F403, F401

import numpy as np
import torch
import torchaudio

from app.config.settings import SAMPLE_RATE, USE_DIARIZATION, HF_TOKEN
from app.utils.logger import logger

# Try to import pyannote.audio, but handle import errors gracefully
# Note: logger might not be initialized yet, so we use print for critical errors
try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
    
    # Patch pyannote.audio's get_model to handle use_auth_token and ensure token is available
    try:
        import functools
        import os
        from pyannote.audio.pipelines.utils.getter import get_model as original_get_model
        
        if not hasattr(original_get_model, '_hf_compat_patched'):
            @functools.wraps(original_get_model)
            def patched_get_model(*args, **kwargs):
                """Convert use_auth_token to token in pyannote's get_model and ensure token is available."""
                # Convert use_auth_token to token
                if 'use_auth_token' in kwargs:
                    token_val = kwargs.pop('use_auth_token')
                    if 'token' not in kwargs:
                        kwargs['token'] = token_val
                
                # If no token in kwargs, try to get it from environment or settings
                if 'token' not in kwargs or kwargs.get('token') is None:
                    # Try environment variables first
                    token_from_env = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
                    if token_from_env:
                        kwargs['token'] = token_from_env
                    # Fallback to settings if available
                    elif HF_TOKEN:
                        kwargs['token'] = HF_TOKEN
                
                return original_get_model(*args, **kwargs)
            
            patched_get_model._hf_compat_patched = True
            
            # Replace in the module
            import pyannote.audio.pipelines.utils.getter as getter_module
            getter_module.get_model = patched_get_model
            import sys
            sys.modules['pyannote.audio.pipelines.utils.getter'].get_model = patched_get_model
    except (ImportError, AttributeError):
        # get_model not available or already patched
        pass
        
except Exception as e:
    PYANNOTE_AVAILABLE = False
    import traceback
    error_msg = f"pyannote.audio import failed: {type(e).__name__}: {e}"
    # Try to log, but if logger isn't ready, print to stderr
    try:
        logger.error(error_msg)
        logger.error(f"Full traceback: {traceback.format_exc()}")
    except:
        import sys
        print(f"ERROR: {error_msg}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

# Global state
diarization_pipeline = None
diarization_loaded = False


def load_diarization_model():
    """Loads the pyannote speaker diarization model."""
    global diarization_pipeline, diarization_loaded
    if diarization_loaded or not USE_DIARIZATION:
        return
    
    if not PYANNOTE_AVAILABLE:
        logger.warning("pyannote.audio is not available. Diarization will be disabled.")
        logger.warning("This may be due to version compatibility issues with torchaudio.")
        diarization_loaded = False
        return
    
    logger.info("Loading pyannote speaker diarization model...")
    logger.info(f"HF_TOKEN: {'Set (length: ' + str(len(HF_TOKEN)) + ')' if HF_TOKEN else 'Not set'}")
    try:
        # Use the speaker diarization model
        # Note: This model requires HuggingFace authentication token
        # Set token as environment variable for huggingface_hub to pick up automatically
        import os
        if HF_TOKEN:
            # Set HF_TOKEN environment variable for huggingface_hub
            os.environ['HF_TOKEN'] = HF_TOKEN
            os.environ['HUGGING_FACE_HUB_TOKEN'] = HF_TOKEN
            # Also try setting it for huggingface_hub's login
            try:
                from huggingface_hub import login
                login(token=HF_TOKEN, add_to_git_credential=False)
            except Exception as e:
                logger.debug(f"Could not login to huggingface_hub: {e}")
        
        # Load pipeline - pass token directly if supported, otherwise rely on environment/login
        # Try passing token as parameter (newer pyannote versions)
        try:
            diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=HF_TOKEN
            )
        except TypeError:
            # Fallback: older pyannote versions might use use_auth_token
            try:
                diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=HF_TOKEN
                )
            except TypeError:
                # Last resort: rely on environment variables and login
                diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1"
                )
        diarization_pipeline.to(torch.device("cpu"))
        diarization_loaded = True
        logger.info("Speaker diarization model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load speaker diarization model: {e}", exc_info=True)
        logger.warning("Note: pyannote models require HuggingFace authentication token.")
        logger.warning("Set HF_TOKEN environment variable with your HuggingFace token.")
        logger.warning("Also ensure you've accepted the model terms at: https://huggingface.co/pyannote/speaker-diarization-3.1")
        diarization_loaded = False


def diarize_audio(audio_float32: np.ndarray) -> list:
    """
    Perform speaker diarization on audio.
    
    Args:
        audio_float32: Audio data as float32 numpy array, normalized to [-1, 1]
    
    Returns:
        List of dicts with 'start', 'end', 'speaker' keys
    """
    if diarization_pipeline is None:
        logger.warning("Diarization model not loaded. Returning empty diarization.")
        return []
    
    try:
        # pyannote expects audio as a dict with 'waveform' and 'sample_rate'
        # Convert numpy array to torch tensor with shape [channels, samples]
        # For mono audio, we need shape [1, samples]
        audio_tensor = torch.from_numpy(audio_float32).unsqueeze(0)  # Shape: [1, samples]
        
        # Run diarization
        diarization = diarization_pipeline({
            "waveform": audio_tensor,
            "sample_rate": SAMPLE_RATE
        })
        
        # Extract segments
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
        
        logger.info(f"Diarization completed: {len(segments)} speaker segments found")
        if segments:
            logger.info(f"Sample diarization segments: {segments[:3]}")
        return segments
        
    except Exception as e:
        logger.error(f"Error during diarization: {e}", exc_info=True)
        import traceback
        logger.error(traceback.format_exc())
        return []


def assign_speakers_to_segments(transcription_segments: list, diarization_segments: list) -> list:
    """
    Assign speaker labels to transcription segments based on diarization.
    
    Args:
        transcription_segments: List of dicts with 'text', 'start', 'end'
        diarization_segments: List of dicts with 'start', 'end', 'speaker'
    
    Returns:
        Transcription segments with added 'speaker' field
    """
    if not diarization_segments:
        # No diarization available, return segments without speaker info
        return transcription_segments
    
    # Create a mapping of time ranges to speakers
    result = []
    diarization_idx = 0
    
    for trans_seg in transcription_segments:
        trans_start = trans_seg["start"]
        trans_end = trans_seg["end"]
        trans_mid = (trans_start + trans_end) / 2.0
        
        # Find the diarization segment that contains the middle of this transcription segment
        speaker = None
        for diar_seg in diarization_segments:
            if diar_seg["start"] <= trans_mid <= diar_seg["end"]:
                speaker = diar_seg["speaker"]
                break
        
        # If no exact match, find the diarization segment with maximum overlap
        if speaker is None:
            max_overlap = 0.0
            best_speaker = None
            for diar_seg in diarization_segments:
                # Calculate overlap
                overlap_start = max(trans_start, diar_seg["start"])
                overlap_end = min(trans_end, diar_seg["end"])
                if overlap_start < overlap_end:
                    overlap = overlap_end - overlap_start
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_speaker = diar_seg["speaker"]
            
            if best_speaker:
                speaker = best_speaker
            else:
                # No overlap found, find the closest diarization segment by distance
                min_distance = float('inf')
                for diar_seg in diarization_segments:
                    # Calculate distance to nearest edge
                    if trans_end < diar_seg["start"]:
                        distance = diar_seg["start"] - trans_end
                    elif trans_start > diar_seg["end"]:
                        distance = trans_start - diar_seg["end"]
                    else:
                        continue  # Should have been caught by overlap check
                    
                    if distance < min_distance:
                        min_distance = distance
                        speaker = diar_seg["speaker"]
        
        # Add speaker to transcription segment
        result_seg = trans_seg.copy()
        result_seg["speaker"] = speaker if speaker else "SPEAKER_UNKNOWN"
        result.append(result_seg)
    
    return result


def is_diarization_loaded() -> bool:
    """Returns whether diarization model is loaded."""
    return diarization_loaded

