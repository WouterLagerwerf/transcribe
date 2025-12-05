# -*- coding: utf-8 -*-
"""
Real-time speaker identification using voice embeddings.

This module provides speaker identification by extracting voice embeddings
and matching them to known speakers. It's designed for real-time streaming
where speakers need to be identified consistently across the entire session.

Key features:
- Consistent speaker IDs across the entire session
- Automatic enrollment of new speakers
- Adaptive voiceprint updates (running average)
- Minimum segment duration requirements for reliable embeddings
- Speaker confirmation (requires consistent matches before assignment)
"""

import os
import warnings
import logging
import contextlib

# Suppress PyTorch Lightning warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*weights_only=False.*")
warnings.filterwarnings("ignore", message=".*ModelCheckpoint.*")
warnings.filterwarnings("ignore", message=".*task-dependent loss.*")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning_fabric").setLevel(logging.ERROR)

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field
from app.utils.logger import logger
from app.config.settings import TORCH_DEVICE, SAMPLE_RATE

# Configuration defaults (can be overridden via environment)
# NOTE: pyannote embedding model produces relatively low cosine similarities (0.4-0.6 for same speaker)
MIN_SEGMENT_DURATION = float(os.getenv("SPEAKER_MIN_SEGMENT_DURATION", "0.8"))  # Minimum seconds for reliable embedding
SIMILARITY_THRESHOLD = float(os.getenv("SPEAKER_SIMILARITY_THRESHOLD", "0.40"))  # Cosine similarity threshold (tuned for pyannote)
ENROLLMENT_THRESHOLD = float(os.getenv("SPEAKER_ENROLLMENT_THRESHOLD", "0.30"))  # Below this = new speaker
ENROLLMENT_FLOOR = float(os.getenv("SPEAKER_ENROLLMENT_FLOOR", "0.35"))  # Do not enroll below this similarity
CONFIRMATION_COUNT = int(os.getenv("SPEAKER_CONFIRMATION_COUNT", "1"))  # Matches needed before confirming speaker
VOICEPRINT_MEMORY = int(os.getenv("SPEAKER_VOICEPRINT_MEMORY", "20"))  # Number of embeddings to remember per speaker
LEARNING_RATE = float(os.getenv("SPEAKER_LEARNING_RATE", "0.15"))  # How fast voiceprints adapt
MIN_EMBED_ENERGY = float(os.getenv("SPEAKER_MIN_ENERGY", "0.001"))  # Minimum average abs energy to accept embedding
ALLOW_UNSAFE_TORCH_LOAD = os.getenv("ALLOW_UNSAFE_TORCH_LOAD", "0")

# Global state
embedding_model = None
embedding_model_loaded = False


def load_speaker_model():
    """Load the speaker embedding model from pyannote."""
    global embedding_model, embedding_model_loaded
    
    if embedding_model_loaded:
        return True
    
    try:
        from pyannote.audio import Model
        from app.config.settings import HF_TOKEN

        logger.info("Loading speaker embedding model (safe weights-only load)...")

        # Allowlist required Lightning callbacks for safe torch.load under PyTorch 2.6
        safe_globals: List[type] = []

        # EarlyStopping
        early_stopping_cls = None
        try:
            from pytorch_lightning.callbacks.early_stopping import EarlyStopping as early_stopping_cls
        except Exception as pl_err:
            try:
                from lightning.pytorch.callbacks.early_stopping import EarlyStopping as early_stopping_cls
            except Exception as lp_err:
                logger.warning(f"Could not import EarlyStopping for safe torch.load: pl_err={pl_err} lp_err={lp_err}")
        if early_stopping_cls:
            safe_globals.append(early_stopping_cls)

        # ModelCheckpoint
        model_checkpoint_cls = None
        try:
            from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint as model_checkpoint_cls
        except Exception as pl_mc_err:
            try:
                from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint as model_checkpoint_cls
            except Exception as lp_mc_err:
                logger.warning(f"Could not import ModelCheckpoint for safe torch.load: pl_mc_err={pl_mc_err} lp_mc_err={lp_mc_err}")
        if model_checkpoint_cls:
            safe_globals.append(model_checkpoint_cls)

        # OmegaConf configs (present in pyannote checkpoint)
        list_config_cls = None
        dict_config_cls = None
        container_metadata_cls = None
        metadata_cls = None
        try:
            from omegaconf.listconfig import ListConfig as list_config_cls
        except Exception as lc_err:
            logger.warning(f"Could not import ListConfig for safe torch.load: {lc_err}")
        try:
            from omegaconf.dictconfig import DictConfig as dict_config_cls
        except Exception as dc_err:
            logger.warning(f"Could not import DictConfig for safe torch.load: {dc_err}")
        try:
            from omegaconf.base import ContainerMetadata as container_metadata_cls
        except Exception as cm_err:
            logger.warning(f"Could not import ContainerMetadata for safe torch.load: {cm_err}")
        try:
            from omegaconf.base import Metadata as metadata_cls
        except Exception as md_err:
            logger.warning(f"Could not import Metadata for safe torch.load: {md_err}")
        # Builtins and typing globals sometimes appear in checkpoints
        builtins_list_cls = list
        from collections import defaultdict as collections_defaultdict
        if list_config_cls:
            safe_globals.append(list_config_cls)
        if dict_config_cls:
            safe_globals.append(dict_config_cls)
        if container_metadata_cls:
            safe_globals.append(container_metadata_cls)
        if metadata_cls:
            safe_globals.append(metadata_cls)
        if builtins_list_cls:
            safe_globals.append(builtins_list_cls)
        if collections_defaultdict:
            safe_globals.append(collections_defaultdict)
        # typing.Any appears in the checkpoint metadata
        try:
            safe_globals.append(Any)
        except Exception as any_err:
            logger.warning(f"Could not add typing.Any to safe globals: {any_err}")
        if not list_config_cls and not dict_config_cls and not container_metadata_cls:
            logger.error("omegaconf is required to load pyannote/embedding safely. Install with `pip install \"omegaconf>=2.3,<3\"` and restart. Speaker ID remains disabled.")
            embedding_model_loaded = False
            return False

        if safe_globals:
            try:
                torch.serialization.add_safe_globals(safe_globals)
                logger.info(f"Allowlisted torch.load safe globals: {[cls.__name__ for cls in safe_globals]}")
            except Exception as sg_err:
                logger.warning(f"Failed to add safe globals for torch.load: {sg_err}")

        # Set token for huggingface_hub
        if HF_TOKEN:
            os.environ["HF_TOKEN"] = HF_TOKEN
            os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

        def _load_model(use_unsafe_weights: bool = False):
            """Load the embedding model, optionally forcing weights_only=False."""
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
                try:
                    return Model.from_pretrained("pyannote/embedding", use_auth_token=HF_TOKEN)
                except TypeError:
                    return Model.from_pretrained("pyannote/embedding")

        try:
            embedding_model_local = _load_model(use_unsafe_weights=False)
            load_mode = "safe"
        except Exception as safe_err:
            logger.error(f"Safe load failed (weights_only=True): {safe_err}", exc_info=True)
            logger.warning("Retrying speaker embedding load with weights_only=False (unsafe). Only enable if you trust the checkpoint source.")
            embedding_model_local = _load_model(use_unsafe_weights=True)
            load_mode = "unsafe_weights_only_false"

        # Move to device and set to eval mode
        embedding_model_local = embedding_model_local.to(torch.device(TORCH_DEVICE))
        embedding_model_local.eval()
        embedding_model = embedding_model_local
        embedding_model_loaded = True

        import sys
        print("", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print("🎤 SPEAKER EMBEDDING MODEL LOADED", file=sys.stderr)
        print(f"   Similarity Threshold: {SIMILARITY_THRESHOLD}", file=sys.stderr)
        print(f"   Min Segment Duration: {MIN_SEGMENT_DURATION}s", file=sys.stderr)
        print(f"   Load Mode: {load_mode}", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print("", file=sys.stderr)

        logger.info(f"✅ Speaker embedding model loaded successfully (mode={load_mode})")
        if load_mode == "unsafe_weights_only_false":
            logger.warning("Model loaded with weights_only=False; ensure checkpoint source is trusted.")
        return True

    except Exception as e:
        logger.error(f"Failed to load speaker embedding model: {e}", exc_info=True)
        logger.warning("Speaker identification will be disabled")
        embedding_model_loaded = False
        return False


def is_model_loaded() -> bool:
    """Check if the speaker embedding model is loaded."""
    return embedding_model_loaded and embedding_model is not None


def extract_embedding(audio_float32: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract speaker embedding from audio segment.
    
    Args:
        audio_float32: Audio data as float32 numpy array, normalized to [-1, 1]
    
    Returns:
        Normalized embedding vector (numpy array) or None if extraction fails
    """
    if embedding_model is None or not embedding_model_loaded:
        logger.warning("Speaker embedding model not loaded")
        return None
    
    # Check minimum duration
    duration = len(audio_float32) / SAMPLE_RATE
    if duration < MIN_SEGMENT_DURATION:
        logger.debug(f"Audio too short for embedding: {duration:.2f}s < {MIN_SEGMENT_DURATION}s")
        return None
    
    # Check if audio has valid content (not silent)
    audio_energy = np.abs(audio_float32).mean()
    if audio_energy < MIN_EMBED_ENERGY:
        logger.debug(f"Audio too quiet for embedding: energy={audio_energy:.6f} < {MIN_EMBED_ENERGY}")
        return None
    
    try:
        # Prepare waveform for pyannote embedding model
        waveform = torch.from_numpy(audio_float32.copy()).float().to(TORCH_DEVICE)
        
        # Try (batch, channel, samples) format first - this is what SincNet expects
        waveform_3d = waveform.unsqueeze(0).unsqueeze(0)  # [1, 1, samples]
        
        logger.info(f"Extracting embedding: duration={duration:.2f}s, shape={waveform_3d.shape}, device={waveform_3d.device}")
        
        # Extract embedding using Model directly  
        with torch.no_grad():
            try:
                embedding = embedding_model(waveform_3d)
                logger.info(f"Model output type: {type(embedding)}, shape: {embedding.shape if hasattr(embedding, 'shape') else 'N/A'}")
            except Exception as e1:
                logger.warning(f"3D shape failed: {e1}, trying 2D...")
                # Fallback to (batch, samples)
                waveform_2d = waveform.unsqueeze(0)  # [1, samples]
                embedding = embedding_model(waveform_2d)
        
        # Check what the model actually returned
        if hasattr(embedding, 'keys'):
            logger.info(f"Model returned dict with keys: {embedding.keys()}")
            # Try to extract the actual embedding
            if 'embedding' in embedding:
                embedding = embedding['embedding']
            elif 'embeddings' in embedding:
                embedding = embedding['embeddings']
        
        # The model returns embeddings of shape (batch, embedding_dim)
        # Convert to numpy
        if isinstance(embedding, torch.Tensor):
            logger.info(f"Raw embedding tensor: shape={embedding.shape}, dtype={embedding.dtype}")
            embedding = embedding.cpu().numpy()
        
        # Flatten and normalize to unit vector
        embedding = embedding.flatten()
        logger.info(f"Flattened embedding: shape={embedding.shape}, pre-norm stats: mean={embedding.mean():.4f}, std={embedding.std():.4f}")
        
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            # Log embedding stats for debugging
            logger.info(f"Normalized embedding: mean={embedding.mean():.4f}, std={embedding.std():.4f}, first5={embedding[:5]}")
            return embedding
        
        logger.warning(f"Embedding has zero norm")
        return None
            
    except Exception as e:
        logger.error(f"Error extracting speaker embedding: {e}", exc_info=True)
        return None


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Calculate cosine similarity between two normalized embeddings."""
    return float(np.dot(emb1, emb2))


@dataclass
class Speaker:
    """Represents an identified speaker with their voiceprint."""
    id: int
    voiceprint: np.ndarray
    embeddings: List[np.ndarray] = field(default_factory=list)
    match_count: int = 0  # Times this speaker has been matched
    confidence_scores: List[float] = field(default_factory=list)
    
    @property
    def label(self) -> str:
        return f"SPEAKER_{self.id:02d}"
    
    @property
    def avg_confidence(self) -> float:
        if not self.confidence_scores:
            return 0.0
        return sum(self.confidence_scores[-10:]) / min(len(self.confidence_scores), 10)
    
    def update_voiceprint(self, new_embedding: np.ndarray, similarity: float):
        """Update the speaker's voiceprint with a new embedding."""
        self.embeddings.append(new_embedding)
        self.confidence_scores.append(similarity)
        self.match_count += 1
        
        # Keep only recent embeddings
        if len(self.embeddings) > VOICEPRINT_MEMORY:
            self.embeddings = self.embeddings[-VOICEPRINT_MEMORY:]
        if len(self.confidence_scores) > VOICEPRINT_MEMORY:
            self.confidence_scores = self.confidence_scores[-VOICEPRINT_MEMORY:]
        
        # Update voiceprint with exponential moving average
        # Higher similarity = more weight to new embedding
        weight = LEARNING_RATE * (0.5 + similarity / 2)  # Scale weight by similarity
        self.voiceprint = (1 - weight) * self.voiceprint + weight * new_embedding
        
        # Re-normalize
        norm = np.linalg.norm(self.voiceprint)
        if norm > 0:
            self.voiceprint = self.voiceprint / norm


class SpeakerIdentifier:
    """
    Manages speaker identification for a streaming session.
    
    Uses voice embeddings to:
    1. Identify speakers from audio segments
    2. Automatically enroll new speakers
    3. Maintain consistent speaker IDs across the session
    4. Update voiceprints as more speech is collected
    """
    
    def __init__(
        self,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        enrollment_threshold: float = ENROLLMENT_THRESHOLD,
        confirmation_count: int = CONFIRMATION_COUNT
    ):
        self.similarity_threshold = similarity_threshold
        self.enrollment_threshold = enrollment_threshold
        self.confirmation_count = confirmation_count
        
        self.speakers: Dict[int, Speaker] = {}
        self.next_speaker_id: int = 0
        
        # Track pending speakers (not yet confirmed)
        self.pending_speakers: Dict[int, Speaker] = {}
        
        # Track recent identifications for smoothing
        self.recent_ids: List[int] = []
        self.last_speaker_id: Optional[int] = None
        self.segments_since_last_id: int = 0
        
        logger.info(
            f"SpeakerIdentifier initialized: "
            f"similarity_threshold={similarity_threshold}, "
            f"enrollment_threshold={enrollment_threshold}, "
            f"confirmation_count={confirmation_count}, "
            f"min_segment_duration={MIN_SEGMENT_DURATION}, "
            f"learning_rate={LEARNING_RATE}"
        )
    
    def identify_speaker(self, audio_float32: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Identify the speaker in an audio segment.
        
        Args:
            audio_float32: Audio data as float32 numpy array, normalized to [-1, 1]
        
        Returns:
            (speaker_label, confidence) tuple
            speaker_label is None if speaker couldn't be identified
            confidence is the similarity score (0.0 to 1.0)
        """
        # Extract embedding
        embedding = extract_embedding(audio_float32)
        if embedding is None:
            audio_energy = np.abs(audio_float32).mean()
            logger.warning(f"Failed to extract embedding: samples={len(audio_float32)}, duration={len(audio_float32)/SAMPLE_RATE:.2f}s, energy={audio_energy:.6f}")
            return (None, 0.0)
        
        # Find best matching speaker
        best_match: Optional[Speaker] = None
        best_similarity: float = 0.0
        second_best_similarity: float = 0.0
        
        all_similarities = []
        
        # Check confirmed speakers
        for speaker in self.speakers.values():
            similarity = cosine_similarity(embedding, speaker.voiceprint)
            all_similarities.append((speaker.label, similarity, "confirmed"))
            if similarity > best_similarity:
                second_best_similarity = best_similarity
                best_similarity = similarity
                best_match = speaker
            elif similarity > second_best_similarity:
                second_best_similarity = similarity
        
        # Check pending speakers too
        for speaker in self.pending_speakers.values():
            similarity = cosine_similarity(embedding, speaker.voiceprint)
            all_similarities.append((speaker.label, similarity, "pending"))
            if similarity > best_similarity:
                second_best_similarity = best_similarity
                best_similarity = similarity
                best_match = speaker
            elif similarity > second_best_similarity:
                second_best_similarity = similarity
        
        # Log all similarities for debugging
        if all_similarities:
            sim_str = ", ".join([f"{s[0]}:{s[1]:.3f}" for s in all_similarities])
            logger.info(f"Speaker similarities: [{sim_str}] → best={best_similarity:.3f}")
        
        # Determine action based on similarity
        if best_match and best_similarity >= self.similarity_threshold:
            # Good match found
            # Check for ambiguity (two speakers with similar scores)
            # With pyannote's lower similarities, use a smaller gap threshold
            gap = best_similarity - second_best_similarity
            if gap < 0.05 and second_best_similarity > 0.35:
                # Ambiguous - but still assign to best match (better than nothing)
                logger.debug(f"Ambiguous but assigning: best={best_similarity:.3f}, gap={gap:.3f}")
            
            # Update speaker voiceprint
            best_match.update_voiceprint(embedding, best_similarity)
            
            # Promote pending speaker to confirmed if enough matches
            if best_match.id in self.pending_speakers:
                if best_match.match_count >= self.confirmation_count:
                    self.speakers[best_match.id] = best_match
                    del self.pending_speakers[best_match.id]
                    logger.info(f"✅ Confirmed speaker: {best_match.label} (after {best_match.match_count} matches)")
            
            self.recent_ids.append(best_match.id)
            if len(self.recent_ids) > 5:
                self.recent_ids.pop(0)
            
            # Track for continuation heuristic
            self.last_speaker_id = best_match.id
            self.segments_since_last_id = 0
            
            return (best_match.label, best_similarity)
        
        elif best_similarity < self.enrollment_threshold:
            # Low similarity to all speakers - likely a new speaker, but only if above floor
            if best_similarity < ENROLLMENT_FLOOR:
                logger.info(f"Similarity {best_similarity:.3f} below enrollment floor {ENROLLMENT_FLOOR:.2f}; skipping enrollment")
                self.segments_since_last_id += 1
                return (None, best_similarity)

            new_speaker = self._enroll_new_speaker(embedding, best_similarity)
            if new_speaker:
                self.last_speaker_id = new_speaker.id
                self.segments_since_last_id = 0
                return (new_speaker.label, best_similarity)
            # If can't enroll, try to match with best pending speaker anyway
            if best_match and best_similarity >= 0.4:
                best_match.update_voiceprint(embedding, best_similarity)
                self.last_speaker_id = best_match.id
                self.segments_since_last_id = 0
                return (best_match.label, best_similarity)
        
        # Similarity is between enrollment and match thresholds
        # Try to match with best speaker if reasonably close (pyannote gives lower similarities)
        if best_match and best_similarity >= 0.35:
            logger.info(f"Marginal match: {best_match.label} with similarity={best_similarity:.3f}")
            best_match.update_voiceprint(embedding, best_similarity)
            self.last_speaker_id = best_match.id
            self.segments_since_last_id = 0
            # Promote pending speaker to confirmed if enough matches
            if best_match.id in self.pending_speakers:
                if best_match.match_count >= self.confirmation_count:
                    self.speakers[best_match.id] = best_match
                    del self.pending_speakers[best_match.id]
                    logger.info(f"✅ Confirmed speaker: {best_match.label} (after {best_match.match_count} matches)")
            return (best_match.label, best_similarity)
        
        # Continuation heuristic: if we recently identified a speaker and this segment
        # is uncertain, assume it's the same speaker (common in conversations)
        if self.last_speaker_id is not None and self.segments_since_last_id < 5:
            last_speaker = self.speakers.get(self.last_speaker_id) or self.pending_speakers.get(self.last_speaker_id)
            if last_speaker and best_similarity >= 0.25:
                logger.info(f"Continuation: assigning to recent speaker {last_speaker.label} (similarity={best_similarity:.3f})")
                last_speaker.update_voiceprint(embedding, best_similarity)
                self.segments_since_last_id += 1
                return (last_speaker.label, best_similarity)
        
        self.segments_since_last_id += 1
        logger.debug(f"Uncertain speaker: similarity={best_similarity:.3f}")
        return (None, best_similarity)
    
    def _enroll_new_speaker(self, embedding: np.ndarray, similarity: Optional[float] = None) -> Optional[Speaker]:
        """Create a new pending speaker."""
        # Check if we already have too many speakers
        total_speakers = len(self.speakers) + len(self.pending_speakers)
        if total_speakers >= 10:
            logger.warning("Maximum speakers reached (10), not enrolling new speaker")
            return None
        
        # Check for rapid enrollment (anti-noise protection)
        # Allow up to 5 pending speakers to handle multi-speaker conversations
        recent_new = sum(1 for s in self.pending_speakers.values() if s.match_count < 2)
        if recent_new >= 5:
            logger.debug("Too many pending speakers, waiting for confirmations")
            return None
        
        speaker_id = self.next_speaker_id
        self.next_speaker_id += 1
        
        speaker = Speaker(
            id=speaker_id,
            voiceprint=embedding.copy(),
            embeddings=[embedding],
            match_count=1
        )
        
        self.pending_speakers[speaker_id] = speaker
        if similarity is not None:
            logger.info(
                f"🎤 New speaker detected: {speaker.label} (pending confirmation) "
                f"(similarity={similarity:.3f} vs thresholds "
                f"match={self.similarity_threshold}, enroll<{self.enrollment_threshold})"
            )
        else:
            logger.info(f"🎤 New speaker detected: {speaker.label} (pending confirmation)")
        
        return speaker
    
    def get_speaker_count(self) -> int:
        """Get number of confirmed speakers."""
        return len(self.speakers)
    
    def get_all_speakers(self) -> List[str]:
        """Get labels of all confirmed speakers."""
        return [s.label for s in sorted(self.speakers.values(), key=lambda x: x.id)]
    
    def get_last_speaker_label(self) -> Optional[str]:
        """Get the label of the most recently identified speaker."""
        if self.last_speaker_id is not None:
            speaker = self.speakers.get(self.last_speaker_id) or self.pending_speakers.get(self.last_speaker_id)
            if speaker:
                return speaker.label
        return None
    
    def get_stats(self) -> Dict:
        """Get statistics about speaker identification."""
        return {
            "confirmed_speakers": len(self.speakers),
            "pending_speakers": len(self.pending_speakers),
            "speakers": [
                {
                    "label": s.label,
                    "match_count": s.match_count,
                    "avg_confidence": s.avg_confidence
                }
                for s in sorted(self.speakers.values(), key=lambda x: x.id)
            ]
        }


def identify_speaker_in_segment(
    audio_float32: np.ndarray,
    identifier: SpeakerIdentifier
) -> Tuple[Optional[str], float]:
    """
    Convenience function to identify speaker in an audio segment.
    
    Args:
        audio_float32: Audio data as float32 numpy array
        identifier: SpeakerIdentifier instance for the session
    
    Returns:
        (speaker_label, confidence) tuple
    """
    return identifier.identify_speaker(audio_float32)

