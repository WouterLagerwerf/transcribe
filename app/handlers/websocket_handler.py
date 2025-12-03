"""WebSocket connection handling and audio processing with speaker identification."""

import asyncio
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
import numpy as np
import websockets

# Use orjson for faster JSON serialization (3-10x faster than stdlib json)
try:
    import orjson
    def json_dumps(obj: Any) -> str:
        """Fast JSON serialization using orjson, returns string for WebSocket text messages."""
        return orjson.dumps(obj).decode('utf-8')
except ImportError:
    import json
    def json_dumps(obj: Any) -> str:
        """Fallback to stdlib json if orjson not available."""
        return json.dumps(obj)

from app.config.settings import (
    SAMPLE_RATE, CHUNK_SIZE_SECONDS, MAX_SEGMENT_SECONDS, USE_DIARIZATION
)
from app.utils.logger import logger
from app.services.transcription import transcribe_synchronous, get_executor, is_server_ready
from app.services.speaker_identification import (
    SpeakerIdentifier, is_model_loaded as is_speaker_model_loaded
)

# Pre-calculated constants
MAX_BUFFER_SAMPLES = int(MAX_SEGMENT_SECONDS * SAMPLE_RATE)


class AudioBuffer:
    """
    Efficient audio buffer using deque to avoid costly np.concatenate on every chunk.
    Each connection gets its own AudioBuffer instance.
    """
    __slots__ = ['_chunks', '_total_samples', '_max_samples']
    
    def __init__(self, max_samples: int = MAX_BUFFER_SAMPLES):
        self._chunks: deque = deque()
        self._total_samples: int = 0
        self._max_samples: int = max_samples
    
    def append(self, audio_chunk: np.ndarray) -> None:
        """Append audio chunk to buffer."""
        self._chunks.append(audio_chunk)
        self._total_samples += len(audio_chunk)
        
        # Trim old chunks if we exceed max size
        while self._total_samples > self._max_samples and self._chunks:
            removed = self._chunks.popleft()
            self._total_samples -= len(removed)
    
    def get_all(self) -> np.ndarray:
        """Get all audio samples."""
        if not self._chunks:
            return np.array([], dtype=np.int16)
        return np.concatenate(list(self._chunks))
    
    def clear(self) -> None:
        """Clear all audio from buffer."""
        self._chunks.clear()
        self._total_samples = 0
    
    def __len__(self) -> int:
        return self._total_samples
    
    @property
    def duration_seconds(self) -> float:
        return self._total_samples / SAMPLE_RATE


@dataclass
class Session:
    """
    Represents a single transcription session (one WebSocket connection).
    All state is isolated per-session to support multi-tenancy.
    """
    session_id: str
    websocket: Any
    audio_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    audio_buffer: AudioBuffer = field(default_factory=AudioBuffer)
    speaker_identifier: Optional[SpeakerIdentifier] = None
    absolute_time_offset: float = 0.0
    segments_processed: int = 0
    
    @property
    def log_prefix(self) -> str:
        """Returns a prefix for log messages including session ID."""
        return f"[{self.session_id[:8]}]"


# Global state for tracking active sessions
active_sessions: Dict[str, Session] = {}


async def identify_speaker_for_segment(
    audio_float32: np.ndarray,
    session: Session,
    executor,
    loop
) -> Optional[str]:
    """
    Identify the speaker for an audio segment using embeddings.
    
    Args:
        audio_float32: Audio data as float32 numpy array
        session: Session instance containing the speaker identifier
        executor: Thread pool executor
        loop: Async event loop
    
    Returns:
        Speaker label (e.g., "SPEAKER_00") or None
    """
    if session.speaker_identifier is None or not is_speaker_model_loaded():
        return None
    
    try:
        # Run speaker identification in thread pool (blocking operation)
        speaker_label, confidence = await loop.run_in_executor(
            executor,
            session.speaker_identifier.identify_speaker,
            audio_float32
        )
        
        if speaker_label:
            logger.debug(f"{session.log_prefix} Identified speaker: {speaker_label} (confidence: {confidence:.3f})")
        
        return speaker_label
        
    except Exception as e:
        logger.warning(f"{session.log_prefix} Speaker identification failed: {e}")
        return None


async def process_transcription(session: Session):
    """
    Process audio chunks and transcribe them with speaker identification.
    
    Each session has its own:
    - Audio buffer (no cross-talk between sessions)
    - Speaker identifier (speakers are tracked per-session)
    - Time offset (timestamps are relative to session start)
    
    Uses faster-whisper's built-in VAD for speech detection.
    Uses speaker embeddings for consistent speaker identification.
    """
    executor = get_executor()
    loop = asyncio.get_running_loop()

    while True:
        try:
            audio_chunk = await session.audio_queue.get()
            
            if audio_chunk is None:  # Sentinel value to stop
                # Process any remaining audio
                buffer_len = len(session.audio_buffer)
                if buffer_len > SAMPLE_RATE * 0.5:  # At least 500ms
                    audio_float32 = session.audio_buffer.get_all().astype(np.float32) / 32768.0
                    
                    # Transcribe
                    segments = await loop.run_in_executor(
                        executor, transcribe_synchronous, audio_float32, session.absolute_time_offset
                    )
                    
                    # Identify speaker for each segment
                    for segment in segments:
                        if segment["text"]:
                            # Extract segment audio for speaker ID
                            seg_start = int((segment["start"] - session.absolute_time_offset) * SAMPLE_RATE)
                            seg_end = int((segment["end"] - session.absolute_time_offset) * SAMPLE_RATE)
                            seg_start = max(0, seg_start)
                            seg_end = min(len(audio_float32), seg_end)
                            
                            if seg_end > seg_start:
                                segment_audio = audio_float32[seg_start:seg_end]
                                speaker = await identify_speaker_for_segment(
                                    segment_audio, session, executor, loop
                                )
                                if speaker:
                                    segment["speaker"] = speaker
                            
                            session.segments_processed += 1
                            await send_transcript(session, segment, is_final=True)
                break

            # Add chunk to buffer
            session.audio_buffer.append(audio_chunk)
            segment_duration_seconds = session.audio_buffer.duration_seconds
            
            # Transcribe when we have enough audio
            if segment_duration_seconds >= CHUNK_SIZE_SECONDS:
                segment_to_transcribe = session.audio_buffer.get_all()
                session.audio_buffer.clear()
                audio_float32 = segment_to_transcribe.astype(np.float32) / 32768.0

                # Transcribe
                segments = await loop.run_in_executor(
                    executor, transcribe_synchronous, audio_float32, session.absolute_time_offset
                )

                # Identify speaker for each segment
                for segment in segments:
                    if segment["text"]:
                        # Extract segment audio for speaker ID
                        seg_start = int((segment["start"] - session.absolute_time_offset) * SAMPLE_RATE)
                        seg_end = int((segment["end"] - session.absolute_time_offset) * SAMPLE_RATE)
                        seg_start = max(0, seg_start)
                        seg_end = min(len(audio_float32), seg_end)
                        
                        if seg_end > seg_start:
                            segment_audio = audio_float32[seg_start:seg_end]
                            speaker = await identify_speaker_for_segment(
                                segment_audio, session, executor, loop
                            )
                            if speaker:
                                segment["speaker"] = speaker
                        
                        session.segments_processed += 1
                        await send_transcript(session, segment, is_final=True)
                
                session.absolute_time_offset += segment_duration_seconds

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"{session.log_prefix} Error in processor: {e}", exc_info=True)
            break


async def send_transcript(session: Session, segment: Dict, is_final: bool = True):
    """Send a transcript segment over the websocket."""
    # Convert numpy types to Python types for JSON serialization
    start_time = float(segment["start"])
    end_time = float(segment["end"])
    
    speaker_info = f" [{segment.get('speaker', 'UNKNOWN')}]" if segment.get('speaker') else ""
    logger.info(f"{session.log_prefix} Transcript: {segment['text']}{speaker_info} [{start_time:.2f}s - {end_time:.2f}s]")
    
    response_data = {
        "session_id": session.session_id,
        "transcript": segment["text"],
        "start": start_time,
        "end": end_time,
        "is_final": is_final
    }
    if segment.get("speaker"):
        response_data["speaker"] = segment["speaker"]
    
    response = json_dumps(response_data)
    try:
        await session.websocket.send(response)
    except (websockets.exceptions.ConnectionClosed, websockets.exceptions.ConnectionClosedOK):
        pass


async def websocket_handler(websocket):
    """
    Handles incoming WebSocket connections.
    
    Each connection gets a unique session with completely isolated state:
    - Unique session ID for tracking/logging
    - Separate audio buffer
    - Separate speaker identifier
    - Separate time tracking
    
    Multiple concurrent connections are fully supported without interference.
    """
    if not is_server_ready():
        await websocket.close(code=1013, reason="Server is not ready, model not loaded.")
        return

    # Create a new session for this connection
    session_id = str(uuid.uuid4())
    session = Session(
        session_id=session_id,
        websocket=websocket,
    )
    
    # Initialize speaker identifier for this session
    if USE_DIARIZATION and is_speaker_model_loaded():
        session.speaker_identifier = SpeakerIdentifier()
        logger.info(f"{session.log_prefix} New session from {websocket.remote_address} - Speaker identification enabled")
    else:
        logger.info(f"{session.log_prefix} New session from {websocket.remote_address} - Speaker identification disabled")

    # Register session
    active_sessions[session_id] = session
    
    # Send session info to client
    try:
        welcome_msg = json_dumps({
            "type": "session_start",
            "session_id": session_id,
            "speaker_identification": session.speaker_identifier is not None
        })
        await websocket.send(welcome_msg)
    except Exception:
        pass

    # Start processing task
    processor_task = asyncio.create_task(process_transcription(session))

    try:
        async for message in websocket:
            if isinstance(message, bytes):
                audio_chunk = np.frombuffer(message, dtype=np.int16)
                await session.audio_queue.put(audio_chunk)

    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception as e:
        logger.error(f"{session.log_prefix} Connection error: {e}", exc_info=True)
    finally:
        # Signal processor to stop
        await session.audio_queue.put(None)
        await processor_task

        # Log session stats before cleanup
        stats = {
            "session_id": session_id,
            "segments_processed": session.segments_processed,
            "total_duration": session.absolute_time_offset,
        }
        if session.speaker_identifier:
            stats["speakers"] = session.speaker_identifier.get_stats()
        
        logger.info(f"{session.log_prefix} Session ended: {stats}")
        
        # Cleanup
        del active_sessions[session_id]


def get_active_sessions() -> Dict[str, Session]:
    """Returns the dict of active sessions."""
    return active_sessions


def get_session_count() -> int:
    """Returns the number of active sessions."""
    return len(active_sessions)
