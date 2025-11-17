# -*- coding: utf-8 -*-
"""Real-time speaker enrollment and identification using voice embeddings."""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from app.utils.logger import logger
from app.config.settings import (
    DEVICE, SAMPLE_RATE,
    ENROLLMENT_SIMILARITY_THRESHOLD, ENROLLMENT_MIN_SEGMENT_DURATION,
    ENROLLMENT_LEARNING_RATE, ENROLLMENT_MIN_CONFIDENCE, ENROLLMENT_ADAPTIVE_THRESHOLD,
    ENROLLMENT_MAX_SPEAKERS, ENROLLMENT_MIN_SAMPLES, ENROLLMENT_CONFIDENCE_WINDOW
)

# Global state
embedding_model = None
embedding_loaded = False


def load_embedding_model():
    """Load the speaker embedding model from pyannote."""
    global embedding_model, embedding_loaded
    
    if embedding_loaded:
        return
    
    try:
        from pyannote.audio import Model
        from app.config.settings import HF_TOKEN
        import os
        
        # Load the embedding model used by pyannote diarization
        # This is the same model used internally for speaker identification
        logger.info("Loading speaker embedding model for voice prints...")
        
        # Set token for huggingface_hub
        if HF_TOKEN:
            os.environ['HF_TOKEN'] = HF_TOKEN
            os.environ['HUGGING_FACE_HUB_TOKEN'] = HF_TOKEN
        
        # Try different ways to load the model
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
        
        embedding_model.to(torch.device(DEVICE))
        embedding_model.eval()
        embedding_loaded = True
        import sys
        print("", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print(f"🎤 SPEAKER EMBEDDING MODEL LOADED", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print("", file=sys.stderr)
        logger.info("✅ Speaker embedding model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}", exc_info=True)
        logger.warning("Speaker enrollment will be disabled")
        embedding_loaded = False


def extract_embedding(audio_float32: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract speaker embedding from audio segment.
    
    Args:
        audio_float32: Audio data as float32 numpy array, normalized to [-1, 1]
    
    Returns:
        Embedding vector (numpy array) or None if extraction fails
    """
    if embedding_model is None or not embedding_loaded:
        return None
    
    try:
        # Convert to torch tensor: [channels, samples]
        # pyannote embedding model expects shape [channels, samples]
        audio_tensor = torch.from_numpy(audio_float32).unsqueeze(0).to(torch.device(DEVICE))
        
        # Extract embedding
        with torch.no_grad():
            # The model expects a dict with 'waveform' and 'sample_rate'
            # Or it might accept tensor directly - try both
            try:
                embedding = embedding_model({
                    'waveform': audio_tensor,
                    'sample_rate': SAMPLE_RATE
                })
            except (TypeError, AttributeError):
                # Try passing tensor directly
                embedding = embedding_model(audio_tensor, SAMPLE_RATE)
            
            # Extract the embedding vector
            if isinstance(embedding, dict):
                embedding = embedding.get('embedding', embedding.get('embeddings', None))
            
            if embedding is None:
                return None
            
            # Convert to numpy and normalize
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.cpu().numpy()
            
            # Handle batch dimension if present
            if len(embedding.shape) > 1:
                # If shape is [batch, features], take first or average
                if embedding.shape[0] == 1:
                    embedding = embedding[0]
                else:
                    # Average across batch dimension
                    embedding = embedding.mean(axis=0)
            
            # Ensure it's 1D
            embedding = embedding.flatten()
            
            # Normalize to unit vector for cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            else:
                return None
            
            return embedding
    except Exception as e:
        logger.error(f"Error extracting embedding: {e}", exc_info=True)
        import traceback
        logger.debug(traceback.format_exc())
        return None


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings."""
    return np.dot(emb1, emb2)


class SpeakerEnrollment:
    """Manages speaker enrollment and identification for a single connection."""
    
    def __init__(self, similarity_threshold: Optional[float] = None):
        """
        Initialize speaker enrollment.
        
        Args:
            similarity_threshold: Minimum cosine similarity to match a speaker (0.0-1.0)
                                  If None, uses ENROLLMENT_SIMILARITY_THRESHOLD from config
        """
        self.base_similarity_threshold = similarity_threshold if similarity_threshold is not None else ENROLLMENT_SIMILARITY_THRESHOLD
        self.similarity_threshold = self.base_similarity_threshold
        self.min_segment_duration = ENROLLMENT_MIN_SEGMENT_DURATION
        self.learning_rate = ENROLLMENT_LEARNING_RATE
        self.min_confidence = ENROLLMENT_MIN_CONFIDENCE
        self.adaptive_threshold = ENROLLMENT_ADAPTIVE_THRESHOLD
        
        # Map: speaker_id -> list of embeddings (for running average)
        self.speaker_embeddings: Dict[int, List[np.ndarray]] = defaultdict(list)
        # Map: speaker_id -> averaged embedding
        self.speaker_voiceprints: Dict[int, np.ndarray] = {}
        # Map: speaker_id -> list of similarity scores (for confidence tracking)
        self.speaker_similarities: Dict[int, List[float]] = defaultdict(list)
        # Next available speaker ID
        self.next_speaker_id = 0
        # Track how many segments per speaker (for averaging)
        self.speaker_counts: Dict[int, int] = defaultdict(int)
        # Track recent enrollments to prevent noise
        self.recent_enrollments: List[Tuple[float, int]] = []  # List of (time, speaker_id) tuples
    
    def _update_adaptive_threshold(self):
        """Update similarity threshold based on number of enrolled speakers."""
        if not self.adaptive_threshold:
            return
        
        num_speakers = len(self.speaker_voiceprints)
        if num_speakers == 0:
            self.similarity_threshold = self.base_similarity_threshold
        elif num_speakers == 1:
            # With 1 speaker, be more lenient to catch variations
            self.similarity_threshold = max(0.65, self.base_similarity_threshold - 0.05)
        elif num_speakers <= 3:
            # With 2-3 speakers, use base threshold
            self.similarity_threshold = self.base_similarity_threshold
        else:
            # With 4+ speakers, be stricter to avoid false matches
            self.similarity_threshold = min(0.85, self.base_similarity_threshold + 0.05)
    
    def identify_speaker(self, embedding: np.ndarray) -> Tuple[int, float]:
        """
        Identify speaker from embedding with improved matching algorithm.
        
        Args:
            embedding: Speaker embedding vector
        
        Returns:
            (speaker_id, similarity_score) tuple
            speaker_id is -1 if no match found (new speaker)
        """
        if len(self.speaker_voiceprints) == 0:
            # No enrolled speakers yet
            return (-1, 0.0)
        
        # Update adaptive threshold
        self._update_adaptive_threshold()
        
        # Find all matches above a lower threshold (for ranking)
        candidate_matches = []
        for speaker_id, voiceprint in self.speaker_voiceprints.items():
            similarity = cosine_similarity(embedding, voiceprint)
            # Consider matches above a lower threshold for ranking
            if similarity >= max(0.5, self.similarity_threshold - 0.15):
                candidate_matches.append((speaker_id, similarity))
        
        if not candidate_matches:
            return (-1, 0.0)
        
        # Sort by similarity (highest first)
        candidate_matches.sort(key=lambda x: x[1], reverse=True)
        best_match_id, best_similarity = candidate_matches[0]
        
        # Check if best match is above threshold
        if best_similarity >= self.similarity_threshold:
            # Check if there's a clear winner (gap between best and second best)
            if len(candidate_matches) > 1:
                second_best_similarity = candidate_matches[1][1]
                gap = best_similarity - second_best_similarity
                # If gap is small (< 0.1), be more conservative
                if gap < 0.1:
                    # Require higher similarity when there's ambiguity
                    if best_similarity < self.similarity_threshold + 0.05:
                        return (-1, best_similarity)
            
            return (best_match_id, best_similarity)
        else:
            # Similarity too low - new speaker
            return (-1, best_similarity)
    
    def enroll_speaker(self, embedding: np.ndarray, speaker_id: Optional[int] = None, similarity: float = 0.0) -> int:
        """
        Enroll a new speaker or update existing speaker's voiceprint.
        
        Args:
            embedding: Speaker embedding vector
            speaker_id: Optional speaker ID to update, or None for new speaker
            similarity: Similarity score for this embedding (used for confidence weighting)
        
        Returns:
            Speaker ID
        """
        import time
        current_time = time.time()
        
        if speaker_id is None:
            # Check if we should enroll (prevent noise enrollment)
            if similarity < self.min_confidence:
                logger.debug(f"Skipping enrollment: similarity {similarity:.3f} below minimum {self.min_confidence:.3f}")
                return -1
            
            # Check for rapid enrollments (might be noise)
            recent_window = 5.0  # 5 seconds
            recent_count = sum(1 for t, _ in self.recent_enrollments if current_time - t < recent_window)
            if recent_count >= 3:
                logger.debug(f"Skipping enrollment: too many recent enrollments ({recent_count} in {recent_window}s)")
                return -1
            
            # New speaker
            speaker_id = self.next_speaker_id
            self.next_speaker_id += 1
            logger.info(f"🎤 Enrolling new speaker: SPEAKER_{speaker_id:02d} (similarity: {similarity:.3f})")
            self.recent_enrollments.append((current_time, speaker_id))
            # Clean old enrollments
            self.recent_enrollments = [(t, sid) for t, sid in self.recent_enrollments if current_time - t < 30.0]
        
        # Add embedding to speaker's collection
        self.speaker_embeddings[speaker_id].append(embedding)
        self.speaker_counts[speaker_id] += 1
        if similarity > 0:
            self.speaker_similarities[speaker_id].append(similarity)
            # Keep only recent similarities
            if len(self.speaker_similarities[speaker_id]) > 20:
                self.speaker_similarities[speaker_id] = self.speaker_similarities[speaker_id][-20:]
        
        # Update voiceprint using adaptive learning rate
        if speaker_id in self.speaker_voiceprints:
            # Update existing voiceprint
            n = self.speaker_counts[speaker_id]
            old_voiceprint = self.speaker_voiceprints[speaker_id]
            
            # Adaptive learning rate: use lower rate for well-established speakers
            if n < 5:
                # Early stages: faster learning
                alpha = min(0.4, self.learning_rate * 1.5)
            elif n < 20:
                # Mid stages: normal learning
                alpha = self.learning_rate
            else:
                # Established speaker: slower learning to maintain stability
                alpha = self.learning_rate * 0.7
            
            # Weight by similarity if available (higher similarity = more weight)
            if similarity > 0 and len(self.speaker_similarities[speaker_id]) > 0:
                avg_similarity = np.mean(self.speaker_similarities[speaker_id])
                # Boost learning rate for high-confidence matches
                if avg_similarity > 0.8:
                    alpha *= 1.2
                elif avg_similarity < 0.7:
                    alpha *= 0.8
            
            self.speaker_voiceprints[speaker_id] = (
                (1 - alpha) * old_voiceprint + alpha * embedding
            )
            # Renormalize
            norm = np.linalg.norm(self.speaker_voiceprints[speaker_id])
            if norm > 0:
                self.speaker_voiceprints[speaker_id] /= norm
        else:
            # First embedding for this speaker
            self.speaker_voiceprints[speaker_id] = embedding.copy()
        
        # Keep only last 50 embeddings per speaker to avoid memory issues
        if len(self.speaker_embeddings[speaker_id]) > 50:
            self.speaker_embeddings[speaker_id] = self.speaker_embeddings[speaker_id][-50:]
        
        return speaker_id
    
    def map_diarization_segments(self, audio_float32: np.ndarray, diarization_segments: List[Dict]) -> Dict[str, int]:
        """
        Map pyannote diarization segments to enrollment speaker IDs.
        This creates a mapping from pyannote speaker labels to our enrollment IDs.
        
        Args:
            audio_float32: Audio segment as float32 numpy array
            diarization_segments: List of diarization segments with 'start', 'end', 'speaker' keys
        
        Returns:
            Dict mapping pyannote speaker labels (e.g., 'SPEAKER_00') to enrollment IDs
        """
        pyannote_to_enrollment = {}
        
        # Group segments by pyannote speaker
        speaker_segments = defaultdict(list)
        for seg in diarization_segments:
            pyannote_speaker = seg.get("speaker")
            if pyannote_speaker:
                speaker_segments[pyannote_speaker].append(seg)
        
        # Process each pyannote speaker's segments
        for pyannote_speaker, segs in speaker_segments.items():
            if pyannote_speaker in pyannote_to_enrollment:
                continue  # Already mapped
            
            # Find non-overlapping segments for this speaker
            non_overlapping = []
            for seg in segs:
                start_idx = int(seg["start"] * SAMPLE_RATE)
                end_idx = int(seg["end"] * SAMPLE_RATE)
                
                # Check if this segment overlaps with other speakers' segments
                overlaps = False
                for other_seg in diarization_segments:
                    if other_seg.get("speaker") != pyannote_speaker:
                        other_start = int(other_seg["start"] * SAMPLE_RATE)
                        other_end = int(other_seg["end"] * SAMPLE_RATE)
                        if not (end_idx <= other_start or start_idx >= other_end):
                            overlaps = True
                            break
                
                if not overlaps and end_idx > start_idx:
                    seg_audio = audio_float32[start_idx:end_idx]
                    if len(seg_audio) > SAMPLE_RATE * self.min_segment_duration:
                        non_overlapping.append(seg_audio)
            
            # Extract embedding from this speaker's segments (average if multiple)
            if non_overlapping:
                embeddings = []
                for seg_audio in non_overlapping:
                    embedding = extract_embedding(seg_audio)
                    if embedding is not None:
                        embeddings.append(embedding)
                
                if embeddings:
                    # Average embeddings from this speaker's segments
                    avg_embedding = np.mean(embeddings, axis=0)
                    norm = np.linalg.norm(avg_embedding)
                    if norm > 0:
                        avg_embedding = avg_embedding / norm
                    
                    # Identify or enroll speaker
                    speaker_id, similarity = self.identify_speaker(avg_embedding)
                    
                    if speaker_id == -1:
                        # New speaker - enroll
                        speaker_id = self.enroll_speaker(avg_embedding, similarity=similarity)
                        logger.info(f"🎤 Enrolled new speaker: pyannote {pyannote_speaker} -> enrollment SPEAKER_{speaker_id:02d} (similarity: {similarity:.3f})")
                    else:
                        # Update existing speaker's voiceprint
                        self.enroll_speaker(avg_embedding, speaker_id, similarity=similarity)
                        logger.debug(f"Matched pyannote {pyannote_speaker} -> enrollment SPEAKER_{speaker_id:02d} (similarity: {similarity:.3f})")
                    
                    pyannote_to_enrollment[pyannote_speaker] = speaker_id
        
        return pyannote_to_enrollment
    
    def process_segment(self, audio_float32: np.ndarray, diarization_segments: Optional[List[Dict]] = None) -> Tuple[int, float]:
        """
        Process audio segment: extract embedding and identify/enroll speaker.
        If diarization segments are provided, processes each non-overlapping segment separately.
        
        Args:
            audio_float32: Audio segment as float32 numpy array
            diarization_segments: Optional list of diarization segments with 'start', 'end', 'speaker' keys
                                 If provided, extracts embeddings from each non-overlapping segment
        
        Returns:
            (speaker_id, similarity_score) tuple
            If multiple speakers detected, returns the most confident match
        """
        # If diarization segments provided, process each speaker segment separately
        if diarization_segments and len(diarization_segments) > 0:
            # Map pyannote speakers to enrollment IDs
            pyannote_to_enrollment = self.map_diarization_segments(audio_float32, diarization_segments)
            
            # Return the first mapped speaker ID (or -1 if none)
            if pyannote_to_enrollment:
                first_enrollment_id = list(pyannote_to_enrollment.values())[0]
                return (first_enrollment_id, 0.8)  # Return a reasonable similarity
        
        # Fallback: process entire segment (for when diarization not available or no non-overlapping segments)
        embedding = extract_embedding(audio_float32)
        if embedding is None:
            return (-1, 0.0)
        
        # Try to identify speaker
        speaker_id, similarity = self.identify_speaker(embedding)
        
        if speaker_id == -1:
            # New speaker - enroll
            speaker_id = self.enroll_speaker(embedding, similarity=similarity)
            if speaker_id >= 0:
                logger.debug(f"New speaker enrolled: SPEAKER_{speaker_id:02d} (similarity: {similarity:.3f})")
        else:
            # Update existing speaker's voiceprint
            self.enroll_speaker(embedding, speaker_id, similarity=similarity)
            logger.debug(f"Speaker identified: SPEAKER_{speaker_id:02d} (similarity: {similarity:.3f})")
        
        return (speaker_id, similarity)
    
    def get_speaker_count(self) -> int:
        """Get the number of enrolled speakers."""
        return len(self.speaker_voiceprints)

