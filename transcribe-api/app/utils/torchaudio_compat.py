"""Compatibility shim for torchaudio with pyannote.audio.

This module MUST be imported before pyannote.audio to fix compatibility issues
with torchaudio 2.0+ which removed several attributes that pyannote.audio expects.
"""

import sys

# Import torchaudio first
import torchaudio

# Fix compatibility issues: torchaudio 2.0+ removed several attributes
# Patch AudioMetaData
if not hasattr(torchaudio, 'AudioMetaData'):
    from typing import NamedTuple
    
    class AudioMetaData(NamedTuple):
        """Compatibility shim for torchaudio.AudioMetaData removed in torchaudio 2.0+."""
        sample_rate: int
        num_frames: int
        num_channels: int
        bits_per_sample: int
        encoding: str
    
    torchaudio.AudioMetaData = AudioMetaData

# Patch list_audio_backends (removed in torchaudio 2.0+)
if not hasattr(torchaudio, 'list_audio_backends'):
    def list_audio_backends():
        """Compatibility shim for torchaudio.list_audio_backends removed in torchaudio 2.0+."""
        # Return a default list - pyannote just checks if the function exists
        return ['soundfile', 'sox']
    
    torchaudio.list_audio_backends = list_audio_backends

# Ensure patches are in sys.modules
if 'torchaudio' in sys.modules:
    if not hasattr(sys.modules['torchaudio'], 'AudioMetaData'):
        sys.modules['torchaudio'].AudioMetaData = torchaudio.AudioMetaData
    if not hasattr(sys.modules['torchaudio'], 'list_audio_backends'):
        sys.modules['torchaudio'].list_audio_backends = torchaudio.list_audio_backends

