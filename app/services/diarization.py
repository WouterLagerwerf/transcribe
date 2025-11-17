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

from app.config.settings import SAMPLE_RATE, USE_DIARIZATION, HF_TOKEN, DEVICE
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
    
    import sys
    print("", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print(f"👥 LOADING SPEAKER DIARIZATION MODEL", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print(f"   Device: {DEVICE.upper()}", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    
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
        
        # Configure pipeline parameters for better speaker separation
        # Make clustering stricter to better distinguish between different speakers
        if hasattr(diarization_pipeline, 'instantiate'):
            # Try to configure clustering parameters for multi-speaker diarization
            try:
                # Access the clustering component and tune it for better separation
                if hasattr(diarization_pipeline, '_pipeline'):
                    # Try to set parameters on the clustering step
                    pipeline_components = diarization_pipeline._pipeline
                    for component_name, component in pipeline_components.items():
                        if 'clustering' in component_name.lower():
                            # Try to configure min/max speakers if the component supports it
                            if hasattr(component, 'min_speakers'):
                                component.min_speakers = 1
                            if hasattr(component, 'max_speakers'):
                                component.max_speakers = 10
                            # Try to make clustering stricter (lower threshold = more separation)
                            # This helps distinguish between similar voices
                            if hasattr(component, 'threshold'):
                                # Lower threshold = more speakers detected (stricter clustering)
                                component.threshold = 0.7  # Default is usually 0.7-0.9, lower = more separation
                                logger.info(f"Set clustering threshold to 0.7 for better speaker separation")
                            if hasattr(component, 'min_cluster_size'):
                                # Smaller min cluster size = more sensitive to speaker differences
                                component.min_cluster_size = 1
                            logger.info(f"Configured {component_name} component for better speaker separation")
            except Exception as e:
                logger.debug(f"Could not configure pipeline parameters: {e}")
        
        diarization_pipeline.to(torch.device(DEVICE))
        diarization_loaded = True
        print("", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print(f"✅ DIARIZATION MODEL LOADED SUCCESSFULLY", file=sys.stderr)
        print(f"   Device: {DEVICE.upper()}", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print("", file=sys.stderr)
        logger.info(f"Speaker diarization model loaded successfully on {DEVICE}.")
    except Exception as e:
        print("", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print(f"⚠️  DIARIZATION MODEL LOADING FAILED", file=sys.stderr)
        print(f"   Error: {e}", file=sys.stderr)
        print(f"   Server will continue without diarization", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print("", file=sys.stderr)
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
        # Move tensor to the same device as the pipeline
        audio_tensor = torch.from_numpy(audio_float32).unsqueeze(0).to(torch.device(DEVICE))  # Shape: [1, samples]
        
        # Run diarization with parameters optimized for 2-speaker scenarios
        # Try to pass min_speakers and max_speakers if supported
        diarization_input = {
            "waveform": audio_tensor,
            "sample_rate": SAMPLE_RATE
        }
        
        # Try to add speaker count hints if the pipeline supports it
        try:
            # Some pyannote versions support num_speakers parameter
            # Don't set a fixed number, let it detect up to 10 speakers
            if hasattr(diarization_pipeline, 'max_speakers'):
                diarization_input["max_speakers"] = 10
        except:
            pass
        
        diarization = diarization_pipeline(diarization_input)
        
        # Extract segments
        # pyannote.audio 3.1+ returns DiarizeOutput which contains an Annotation object
        segments = []
        
        # Log the type for debugging
        logger.debug(f"Diarization output type: {type(diarization)}")
        logger.debug(f"Diarization attributes: {[a for a in dir(diarization) if not a.startswith('_')]}")
        
        try:
            # Try to get the Annotation object from DiarizeOutput
            annotation = None
            
            # Check various possible attribute names for the annotation
            for attr_name in ['annotation', 'speaker_diarization', 'diarization']:
                if hasattr(diarization, attr_name):
                    attr_value = getattr(diarization, attr_name)
                    logger.debug(f"Found attribute '{attr_name}': {type(attr_value)}")
                    # Check if this attribute has itertracks (it's an Annotation)
                    if hasattr(attr_value, 'itertracks'):
                        annotation = attr_value
                        logger.debug(f"Using '{attr_name}' as annotation")
                        break
            
            # If no annotation found via attributes, check if diarization itself is an Annotation
            if annotation is None:
                if hasattr(diarization, 'itertracks'):
                    annotation = diarization
                    logger.debug("Using diarization directly as annotation")
                elif isinstance(diarization, dict):
                    # Try dict access
                    for key in ['annotation', 'speaker_diarization', 'diarization']:
                        if key in diarization and hasattr(diarization[key], 'itertracks'):
                            annotation = diarization[key]
                            logger.debug(f"Using dict key '{key}' as annotation")
                            break
            
            # If still no annotation, try importing Annotation and checking type
            if annotation is None:
                from pyannote.core import Annotation
                if isinstance(diarization, Annotation):
                    annotation = diarization
                    logger.debug("Diarization is an Annotation instance")
            
            # If we found an annotation, iterate over it
            if annotation is not None:
                for segment, track, label in annotation.itertracks(yield_label=True):
                    segments.append({
                        "start": segment.start,
                        "end": segment.end,
                        "speaker": label
                    })
            else:
                # Last resort: try to access segments directly
                logger.warning("Could not find annotation attribute, trying direct access...")
                if hasattr(diarization, 'get_timeline'):
                    timeline = diarization.get_timeline()
                    for segment in timeline:
                        label = diarization[segment]
                        segments.append({
                            "start": segment.start,
                            "end": segment.end,
                            "speaker": str(label) if label else "SPEAKER_00"
                        })
                else:
                    raise AttributeError(f"Cannot find annotation in diarization output. Type: {type(diarization)}, Attributes: {[a for a in dir(diarization) if not a.startswith('_')]}")
                    
        except Exception as e:
            logger.error(f"Error extracting diarization segments: {e}", exc_info=True)
            logger.error(f"Diarization type: {type(diarization)}")
            logger.error(f"Diarization repr: {repr(diarization)}")
            raise
        
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

