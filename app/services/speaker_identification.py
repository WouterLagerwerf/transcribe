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

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field
from app.utils.logger import logger
from app.config.settings import TORCH_DEVICE, SAMPLE_RATE

# Configuration defaults (can be overridden via environment)
import os
MIN_SEGMENT_DURATION = float(os.getenv("SPEAKER_MIN_SEGMENT_DURATION", "0.5"))  # Minimum seconds for reliable embedding
SIMILARITY_THRESHOLD = float(os.getenv("SPEAKER_SIMILARITY_THRESHOLD", "0.70"))  # Cosine similarity threshold
ENROLLMENT_THRESHOLD = float(os.getenv("SPEAKER_ENROLLMENT_THRESHOLD", "0.65"))  # Threshold for new speaker enrollment
CONFIRMATION_COUNT = int(os.getenv("SPEAKER_CONFIRMATION_COUNT", "2"))  # Matches needed before confirming speaker
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
        
        # Load the embedding model
        try:
            embedding_model = Model.from_pretrained(
                "pyannote/embedding",
                token=HF_TOKEN
            )
        except TypeError:
            try:
                embedding_model = Model.from_pretrained(
                    "pyannote/embedding",
                    use_auth_token=HF_TOKEN
                )
            except TypeError:
                embedding_model = Model.from_pretrained("pyannote/embedding")
        
        embedding_model.to(TORCH_DEVICE)
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
    return embedding_model_loaded


def extract_embedding(audio_float32: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract speaker embedding from audio segment.
    
    Args:
        audio_float32: Audio data as float32 numpy array, normalized to [-1, 1]
    
    Returns:
        Normalized embedding vector (numpy array) or None if extraction fails
    """
    if embedding_model is None or not embedding_model_loaded:
        return None
    
    # Check minimum duration
    duration = len(audio_float32) / SAMPLE_RATE
    if duration < MIN_SEGMENT_DURATION:
        logger.debug(f"Audio too short for embedding: {duration:.2f}s < {MIN_SEGMENT_DURATION}s")
        return None
    
    try:
        # Convert to torch tensor: [1, samples] for single channel
        audio_tensor = torch.from_numpy(audio_float32).unsqueeze(0).to(TORCH_DEVICE)
        
        with torch.no_grad():
            # Try dict input first (newer pyannote versions)
            try:
                result = embedding_model({
                    'waveform': audio_tensor,
                    'sample_rate': SAMPLE_RATE
                })
            except (TypeError, AttributeError):
                # Fallback to direct tensor input
                result = embedding_model(audio_tensor, SAMPLE_RATE)
            
            # Extract embedding from result
            if isinstance(result, dict):
                embedding = result.get('embedding', result.get('embeddings', None))
            else:
                embedding = result
            
            if embedding is None:
                return None
            
            # Convert to numpy
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.cpu().numpy()
            
            # Handle batch dimension
            if len(embedding.shape) > 1:
                if embedding.shape[0] == 1:
                    embedding = embedding[0]
                else:
                    # Average across batch
                    embedding = embedding.mean(axis=0)
            
            # Flatten and normalize to unit vector
            embedding = embedding.flatten()
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                return embedding
            
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
            return (None, 0.0)
        
        # Find best matching speaker
        best_match: Optional[Speaker] = None
        best_similarity: float = 0.0
        second_best_similarity: float = 0.0
        
        # Check confirmed speakers
        for speaker in self.speakers.values():
            similarity = cosine_similarity(embedding, speaker.voiceprint)
            if similarity > best_similarity:
                second_best_similarity = best_similarity
                best_similarity = similarity
                best_match = speaker
            elif similarity > second_best_similarity:
                second_best_similarity = similarity
        
        # Check pending speakers too
        for speaker in self.pending_speakers.values():
            similarity = cosine_similarity(embedding, speaker.voiceprint)
            if similarity > best_similarity:
                second_best_similarity = best_similarity
                best_similarity = similarity
                best_match = speaker
            elif similarity > second_best_similarity:
                second_best_similarity = similarity
        
        # Determine action based on similarity
        if best_match and best_similarity >= self.similarity_threshold:
            # Good match found
            # Check for ambiguity (two speakers with similar scores)
            gap = best_similarity - second_best_similarity
            if gap < 0.08 and best_similarity < self.similarity_threshold + 0.1:
                # Ambiguous - don't assign
                logger.debug(f"Ambiguous speaker match: best={best_similarity:.3f}, gap={gap:.3f}")
                return (None, best_similarity)
            
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
            
            return (best_match.label, best_similarity)
        
        elif best_similarity < self.enrollment_threshold:
            # Low similarity to all speakers - likely a new speaker
            new_speaker = self._enroll_new_speaker(embedding)
            if new_speaker:
                return (new_speaker.label, best_similarity)
        
        # Similarity is between enrollment and match thresholds
        # Could be a new speaker or a poor quality sample
        # Be conservative and don't assign
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
        recent_new = sum(1 for s in self.pending_speakers.values() if s.match_count < 2)
        if recent_new >= 3:
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

