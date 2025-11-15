"""Whisper transcription API application package."""

# CRITICAL: Import compatibility shims FIRST before any other imports
# These patches are required for pyannote.audio to work with newer library versions

# 1. Patch huggingface_hub to accept use_auth_token (must be before pyannote imports hf_hub)
from app.utils.hf_hub_compat import *  # noqa: F403, F401

# 2. Patch torchaudio.AudioMetaData which pyannote.audio requires
from app.utils.torchaudio_compat import *  # noqa: F403, F401

