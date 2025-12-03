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

# Suppress PyTorch Lightning warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*weights_only=False.*")
warnings.filterwarnings("ignore", message=".*ModelCheckpoint.*")
warnings.filterwarnings("ignore", message=".*task-dependent loss.*")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning_fabric").setLevel(logging.ERROR)

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field
from app.utils.logger import logger
from app.config.settings import TORCH_DEVICE, SAMPLE_RATE

# Configuration defaults (can be overridden via environment)
# NOTE: pyannote embedding model produces relatively low cosine similarities (0.4-0.6 for same speaker)
MIN_SEGMENT_DURATION = float(os.getenv("SPEAKER_MIN_SEGMENT_DURATION", "0.3"))  # Minimum seconds for reliable embedding
SIMILARITY_THRESHOLD = float(os.getenv("SPEAKER_SIMILARITY_THRESHOLD", "0.40"))  # Cosine similarity threshold (tuned for pyannote)
ENROLLMENT_THRESHOLD = float(os.getenv("SPEAKER_ENROLLMENT_THRESHOLD", "0.30"))  # Below this = new speaker
CONFIRMATION_COUNT = int(os.getenv("SPEAKER_CONFIRMATION_COUNT", "1"))  # Matches needed before confirming speaker
VOICEPRINT_MEMORY = int(os.getenv("SPEAKER_VOICEPRINT_MEMORY", "20"))  # Number of embeddings to remember per speaker
LEARNING_RATE = float(os.getenv("SPEAKER_LEARNING_RATE", "0.15"))  # How fast voiceprints adapt

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
        
        logger.info("Loading speaker embedding model...")
        
        # Set token for huggingface_hub
        if HF_TOKEN:
            os.environ['HF_TOKEN'] = HF_TOKEN
            os.environ['HUGGING_FACE_HUB_TOKEN'] = HF_TOKEN
        
        # Load the model directly (not through Inference wrapper)
        try:
            embedding_model = Model.from_pretrained(
                "pyannote/embedding",
                use_auth_token=HF_TOKEN
            )
        except TypeError:
            embedding_model = Model.from_pretrained("pyannote/embedding")
        
        # Move to device and set to eval mode
        embedding_model = embedding_model.to(torch.device(TORCH_DEVICE))
        embedding_model.eval()
        embedding_model_loaded = True
        
        import sys
        print("", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print("🎤 SPEAKER EMBEDDING MODEL LOADED", file=sys.stderr)
        print(f"   Similarity Threshold: {SIMILARITY_THRESHOLD}", file=sys.stderr)
        print(f"   Min Segment Duration: {MIN_SEGMENT_DURATION}s", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print("", file=sys.stderr)
        
        logger.info("✅ Speaker embedding model loaded successfully")
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
    if audio_energy < 0.0001:  # Lowered threshold - allow quieter audio
        logger.debug(f"Audio too quiet for embedding: energy={audio_energy:.6f}")
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
        
        logger.info(f"SpeakerIdentifier initialized: threshold={similarity_threshold}, confirm={confirmation_count}")
    
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
            # Low similarity to all speakers - likely a new speaker
            new_speaker = self._enroll_new_speaker(embedding)
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
    
    def _enroll_new_speaker(self, embedding: np.ndarray) -> Optional[Speaker]:
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

