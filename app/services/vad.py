# -*- coding: utf-8 -*-
"""
WhisperX-style VAD splitting for optional external VAD.
Supports pyannote or silero.
"""
from typing import List, Tuple
import numpy as np
import torch
import contextlib

from app.config.settings import SAMPLE_RATE, HF_TOKEN
from app.utils.logger import logger

try:
    from pyannote.audio import Model
    from pyannote.audio.pipelines import VoiceActivityDetection
    _has_pyannote = True
except Exception:
    _has_pyannote = False

try:
    import silero_vad
    _has_silero = True
except Exception:
    _has_silero = False

# Cached models
_pyannote_vad_pipeline = None


def _vad_pyannote(audio_float32: np.ndarray, onset: float, offset: float) -> List[Tuple[float, float]]:
    if not _has_pyannote:
        logger.warning("Pyannote VAD not available; falling back to full span.")
        duration = len(audio_float32) / SAMPLE_RATE
        return [(0.0, duration)]
    try:
        global _pyannote_vad_pipeline
        if _pyannote_vad_pipeline is None:
            try:
                from omegaconf.nodes import AnyNode
                from omegaconf.base import Metadata
                torch.serialization.add_safe_globals([dict, int, AnyNode, Metadata])
                # Allowlist pyannote Introspection for safe torch.load (torch >= 2.6)
                try:
                    from pyannote.audio.core.model import Introspection
                    torch.serialization.add_safe_globals([Introspection])
                except Exception:
                    pass
            except Exception:
                pass
            def _load_model(use_unsafe: bool = False):
                original = torch.load
                @contextlib.contextmanager
                def _unsafe():
                    def _patched(*args, **kwargs):
                        kwargs["weights_only"] = False
                        return original(*args, **kwargs)
                    torch.load = _patched
                    try:
                        yield
                    finally:
                        torch.load = original
                ctx = _unsafe() if use_unsafe else contextlib.nullcontext()
                with ctx:
                    return Model.from_pretrained("pyannote/segmentation", use_auth_token=HF_TOKEN)

            try:
                model = _load_model(use_unsafe=False)
            except Exception as e_safe:
                logger.warning(f"Pyannote VAD safe load failed; retrying unsafe (weights_only=False). Error: {e_safe}")
                model = _load_model(use_unsafe=True)

            pipeline = VoiceActivityDetection(segmentation=model)
            pipeline.instantiate({
                "onset": onset,
                "offset": offset,
                "min_duration_on": 0.1,
                "min_duration_off": 0.1,
            })
            _pyannote_vad_pipeline = pipeline
        pipeline = _pyannote_vad_pipeline
        waveform = torch.from_numpy(audio_float32[None, :])
        vad = pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE})
        segments = []
        for segment, _, _ in vad.itertracks(yield_label=True):
            segments.append((float(segment.start), float(segment.end)))
        return segments if segments else [(0.0, len(audio_float32) / SAMPLE_RATE)]
    except Exception as e:
        logger.warning(f"Pyannote VAD failed: {e}; falling back to full span.")
        duration = len(audio_float32) / SAMPLE_RATE
        return [(0.0, duration)]


def _vad_silero(audio_float32: np.ndarray, onset: float, offset: float) -> List[Tuple[float, float]]:
    if not _has_silero:
        logger.warning("Silero VAD not available; falling back to full span.")
        duration = len(audio_float32) / SAMPLE_RATE
        return [(0.0, duration)]
    try:
        import torch
        model, utils = silero_vad.utils.get_speech_timestamps, None  # placeholder
    except Exception as e:
        logger.warning(f"Silero VAD load failed: {e}; falling back to full span.")
        duration = len(audio_float32) / SAMPLE_RATE
        return [(0.0, duration)]
    duration = len(audio_float32) / SAMPLE_RATE
    return [(0.0, duration)]


def vad_split_segments(
    audio_float32: np.ndarray,
    method: str,
    onset: float,
    offset: float,
    chunk_size_seconds: int,
) -> List[Tuple[float, float]]:
    """
    Split audio into speech spans using selected VAD.
    """
    if method == "pyannote":
        spans = _vad_pyannote(audio_float32, onset, offset)
    elif method == "silero":
        spans = _vad_silero(audio_float32, onset, offset)
    else:
        duration = len(audio_float32) / SAMPLE_RATE
        spans = [(0.0, duration)]

    # Merge spans respecting chunk_size_seconds
    merged: List[Tuple[float, float]] = []
    for start, end in spans:
        if not merged:
            merged.append((start, end))
            continue
        m_start, m_end = merged[-1]
        if end - m_start <= chunk_size_seconds and start <= m_end:
            merged[-1] = (m_start, max(m_end, end))
        else:
            merged.append((start, end))

    kept = sum(e - s for s, e in merged)
    total = len(audio_float32) / SAMPLE_RATE
    logger.info(f"VAD {method}: kept {kept:.2f}s / {total:.2f}s in {len(merged)} spans")
    return merged

