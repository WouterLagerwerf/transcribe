# -*- coding: utf-8 -*-
"""
Overlap-capable speaker diarization service.

Uses a pyannote diarization pipeline to produce time-stamped speaker regions
that can overlap. Regions are later mapped to session-level speaker identities
by the speaker identification module.
"""

import os
import logging
import warnings
import contextlib
from typing import List, Dict, Any

import numpy as np
import torch

from app.config.settings import TORCH_DEVICE, SAMPLE_RATE
from app.utils.logger import logger

# Silence noisy warnings from PyTorch Lightning when loading pipelines
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*weights_only=False.*")
warnings.filterwarnings("ignore", message=".*ModelCheckpoint.*")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning_fabric").setLevel(logging.ERROR)

# Environment-configurable diarization model
DIARIZATION_MODEL_NAME = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")
MIN_SPEAKERS = os.getenv("DIARIZATION_MIN_SPEAKERS", None)
MAX_SPEAKERS = os.getenv("DIARIZATION_MAX_SPEAKERS", None)
NUM_SPEAKERS = os.getenv("DIARIZATION_NUM_SPEAKERS", None)
ALLOW_UNSAFE_TORCH_LOAD = os.getenv("ALLOW_UNSAFE_TORCH_LOAD", "0")

diarization_pipeline = None
diarization_model_loaded = False


def load_diarization_model() -> bool:
    """Load the overlap-aware diarization pipeline."""
    global diarization_pipeline, diarization_model_loaded

    if diarization_model_loaded and diarization_pipeline is not None:
        return True

    try:
        from pyannote.audio import Pipeline
        from huggingface_hub import login
        from app.config.settings import HF_TOKEN

        logger.info(f"Loading diarization pipeline: model={DIARIZATION_MODEL_NAME}")

        # Add safe globals required by the pipeline checkpoint
        safe_globals = []
        try:
            from torch.torch_version import TorchVersion
            safe_globals.append(TorchVersion)
        except Exception as tv_err:
            logger.warning(f"Could not import TorchVersion for safe torch.load: {tv_err}")
        # pyannote pipeline checkpoints reference Specifications
        try:
            from pyannote.audio.core.task import Specifications
            safe_globals.append(Specifications)
        except Exception as spec_err:
            logger.warning(f"Could not import Specifications for safe torch.load: {spec_err}")
        try:
            from pyannote.audio.core.task import Problem
            safe_globals.append(Problem)
        except Exception as prob_err:
            logger.warning(f"Could not import Problem for safe torch.load: {prob_err}")
        try:
            from pyannote.audio.core.task import Resolution
            safe_globals.append(Resolution)
        except Exception as res_err:
            logger.warning(f"Could not import Resolution for safe torch.load: {res_err}")
        try:
            from collections import defaultdict as collections_defaultdict
            safe_globals.append(collections_defaultdict)
        except Exception as dd_err:
            logger.warning(f"Could not import collections.defaultdict for safe torch.load: {dd_err}")
        # Builtins may appear in pickled metadata
        safe_globals.append(list)

        if safe_globals:
            try:
                torch.serialization.add_safe_globals(safe_globals)
            except Exception as sg_err:
                logger.warning(f"Failed to add safe globals for torch.load: {sg_err}")

        if HF_TOKEN:
            os.environ["HF_TOKEN"] = HF_TOKEN
            os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
            try:
                login(token=HF_TOKEN, add_to_git_credential=False)
            except Exception as login_err:
                logger.warning(f"HuggingFace login failed: {login_err}")

        def _load_pipeline(use_unsafe_weights: bool = False):
            original_torch_load = torch.load

            @contextlib.contextmanager
            def _unsafe_torch_load():
                def _patched(*args, **kwargs):
                    kwargs["weights_only"] = False
                    return original_torch_load(*args, **kwargs)

                torch.load = _patched
                try:
                    yield
                finally:
                    torch.load = original_torch_load

            load_ctx = _unsafe_torch_load() if use_unsafe_weights else contextlib.nullcontext()
            with load_ctx:
                return Pipeline.from_pretrained(DIARIZATION_MODEL_NAME)

        try:
            diarization_pipeline_local = _load_pipeline(use_unsafe_weights=False)
            load_mode = "safe"
        except Exception as safe_err:
            logger.error(f"Safe diarization load failed (weights_only=True): {safe_err}", exc_info=True)
            logger.warning("Retrying diarization load with weights_only=False (unsafe). Only enable if you trust the checkpoint source.")
            diarization_pipeline_local = _load_pipeline(use_unsafe_weights=True)
            load_mode = "unsafe_weights_only_false"

        # Move to configured device if supported
        try:
            diarization_pipeline_local.to(TORCH_DEVICE)
        except Exception as move_err:
            logger.debug(f"Diarization pipeline .to({TORCH_DEVICE}) not applied: {move_err}")

        diarization_pipeline = diarization_pipeline_local
        diarization_model_loaded = True

        logger.info(f"✅ Diarization pipeline loaded successfully (mode={load_mode})")
        return True

    except Exception as e:
        logger.error(f"Failed to load diarization pipeline: {e}", exc_info=True)
        diarization_model_loaded = False
        diarization_pipeline = None
        return False


def is_diarization_model_loaded() -> bool:
    """Return True if the diarization pipeline is loaded."""
    return diarization_model_loaded and diarization_pipeline is not None


def diarize_audio(audio_float32: np.ndarray) -> List[Dict[str, Any]]:
    """
    Run diarization on an audio chunk and return overlap-capable regions.

    Args:
        audio_float32: Mono audio at SAMPLE_RATE normalized to [-1, 1]

    Returns:
        List of dicts: [{"start": float, "end": float, "label": str}]
        Labels are the diarization pipeline speaker tags (not session IDs).
    """
    if not is_diarization_model_loaded():
        logger.debug("Diarization pipeline not loaded; skipping diarization")
        return []

    if len(audio_float32) == 0:
        return []

    # Basic SNR/energy gate to skip very quiet audio
    energy = float(np.abs(audio_float32).mean())
    if energy < 0.0005:
        logger.info(f"Diarization skipped due to low energy: {energy:.6f}")
        return []

    try:
        waveform = torch.from_numpy(audio_float32).float().unsqueeze(0)  # [1, samples]
        diarization = diarization_pipeline(
            {"waveform": waveform, "sample_rate": SAMPLE_RATE},
            num_speakers=int(NUM_SPEAKERS) if NUM_SPEAKERS else None,
            min_speakers=int(MIN_SPEAKERS) if MIN_SPEAKERS else None,
            max_speakers=int(MAX_SPEAKERS) if MAX_SPEAKERS else None,
        )

        regions: List[Dict[str, Any]] = []
        # pyannote.audio >=3.1 returns Diarization / DiarizeOutput, not Annotation
        if hasattr(diarization, "itertracks"):
            logger.debug("Diarization output has itertracks()")
            for segment, _, speaker in diarization.itertracks(yield_label=True):
                regions.append({
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "label": str(speaker)
                })
        elif hasattr(diarization, "get_timeline"):
            # Newer API: diarization has .get_timeline() and is iterable
            logger.debug("Diarization output has get_timeline(); using itertracks()")
            for segment, track, speaker in diarization.itertracks(yield_label=True):
                regions.append({
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "label": str(speaker)
                })
        elif isinstance(diarization, dict) and "segmentations" in diarization:
            # Fallback for newer DiarizeOutput structure
            # Expect diarization["speakers"] and diarization["segments"]
            logger.debug("Diarization output is dict; using segments/speakers keys")
            segments = diarization.get("segments", [])
            speakers = diarization.get("speakers", [])
            for seg in segments:
                regions.append({
                    "start": float(seg.get("start", 0.0)),
                    "end": float(seg.get("end", 0.0)),
                    "label": str(speakers[seg.get("speaker", 0)]) if speakers else str(seg.get("speaker", "UNK"))
                })

        logger.info(f"Diarization produced {len(regions)} regions")
        regions.sort(key=lambda r: r["start"])
        return regions

    except Exception as e:
        logger.error(f"Error during diarization: {e}", exc_info=True)
        return []

