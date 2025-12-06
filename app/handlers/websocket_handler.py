"""WebSocket connection handling and audio processing with speaker identification."""

import asyncio
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
import numpy as np
import websockets
import torch

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
    SAMPLE_RATE,
    STREAM_WINDOW_SECONDS,
    STREAM_HOP_SECONDS,
    STREAM_FINAL_LAG_SECONDS,
    STREAM_PARTIAL_DEBOUNCE_MS,
    STREAM_ENABLE_PARTIALS,
    MAX_SEGMENT_SECONDS,
    USE_DIARIZATION,
    USE_ALIGNMENT,
    ALIGN_MODEL_NAME,
    LANGUAGE,
    TORCH_DEVICE,
    DIAR_ROLLING_SECONDS,
    DIAR_HOP_SECONDS,
    AUDIO_QUEUE_MAXSIZE,
)
from app.utils.logger import logger
from app.services.transcription import transcribe_synchronous, get_executor, is_server_ready
from app.services.speaker_identification import (
    SpeakerIdentifier, is_model_loaded as is_speaker_model_loaded
)
from app.services.alignment import align_segments
from app.services.diarization import (
    diarize_audio,
    is_diarization_model_loaded
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
    
    def append(self, audio_chunk: np.ndarray) -> int:
        """
        Append audio chunk to buffer.
        Returns number of samples trimmed from the front (for time tracking).
        """
        self._chunks.append(audio_chunk)
        self._total_samples += len(audio_chunk)
        
        removed_samples = 0
        # Trim old chunks if we exceed max size
        while self._total_samples > self._max_samples and self._chunks:
            removed = self._chunks.popleft()
            removed_samples += len(removed)
            self._total_samples -= len(removed)
        
        return removed_samples
    
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
    audio_queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=AUDIO_QUEUE_MAXSIZE))
    audio_buffer: AudioBuffer = field(default_factory=AudioBuffer)
    diar_buffer: AudioBuffer = field(default_factory=lambda: AudioBuffer(max_samples=int(DIAR_ROLLING_SECONDS * SAMPLE_RATE)))
    speaker_identifier: Optional[SpeakerIdentifier] = None
    buffer_start_time: float = 0.0  # absolute time of the first sample in audio_buffer
    diar_buffer_start_time: float = 0.0  # absolute time of the first sample in diar_buffer
    next_window_start: float = 0.0
    last_partial_sent_at: float = 0.0
    last_diar_run: float = 0.0
    diarization_cache: List[Dict[str, Any]] = field(default_factory=list)
    pending_segments: List[Dict[str, Any]] = field(default_factory=list)
    last_known_speaker: Optional[str] = None
    absolute_time_offset: float = 0.0  # kept for stats compatibility
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
    if session.speaker_identifier is None:
        logger.debug(f"{session.log_prefix} No speaker identifier for session")
        return None
    
    if not is_speaker_model_loaded():
        logger.warning(f"{session.log_prefix} Speaker model not loaded")
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
        else:
            logger.debug(f"{session.log_prefix} identify_speaker returned None")
        
        return speaker_label
        
    except Exception as e:
        logger.error(f"{session.log_prefix} Speaker identification failed: {e}", exc_info=True)
        return None


async def map_diarization_to_session_speakers(
    diar_segments: list,
    audio_float32: np.ndarray,
    session: Session,
    executor,
    loop
) -> list:
    """Map diarization regions to session speaker labels using embeddings."""
    if not diar_segments:
        logger.info(f"{session.log_prefix} No diarization regions to map")
        return []
    if session.speaker_identifier is None or not is_speaker_model_loaded():
        return []

    mapped = []
    for region in diar_segments:
        seg_start = max(0, int(region["start"] * SAMPLE_RATE))
        seg_end = min(len(audio_float32), int(region["end"] * SAMPLE_RATE))
        if seg_end <= seg_start:
            continue

        segment_audio = audio_float32[seg_start:seg_end]
        try:
            speaker_label, confidence = await loop.run_in_executor(
                executor,
                session.speaker_identifier.identify_speaker,
                segment_audio
            )
        except Exception as e:
            logger.error(f"{session.log_prefix} Diarization mapping failed: {e}", exc_info=True)
            continue

        if not speaker_label and session.speaker_identifier:
            speaker_label = session.speaker_identifier.get_last_speaker_label()

        if speaker_label:
            mapped.append({
                "start": region["start"],
                "end": region["end"],
                "speaker": speaker_label,
                "score": confidence
            })
    logger.info(
        f"{session.log_prefix} Diarization mapped {len(mapped)}/{len(diar_segments)} regions to session speakers"
    )
    return mapped


async def run_transcribe_and_diarize(
    audio_float32: np.ndarray,
    session: Session,
    time_offset: float,
    executor,
    loop,
    diarization_override: Optional[list] = None
) -> Dict[str, Any]:
    """
    Run transcription and diarization in parallel for an audio chunk.

    Returns a dict with:
      {
        "segments": [...],  # transcription segments
        "diarization": [...],  # diarization regions with session speaker labels
      }
    """
    diar_regions = diarization_override or []

    diar_task = None
    if diarization_override is None and session.speaker_identifier and is_diarization_model_loaded():
        diar_task = loop.run_in_executor(executor, diarize_audio, audio_float32)

    trans_task = loop.run_in_executor(
        executor, transcribe_synchronous, audio_float32, time_offset
    )

    if diar_task:
        diar_regions = await diar_task

    segments = await trans_task

    # Optional alignment (post-ASR) if enabled
    try:
        if USE_ALIGNMENT:
            align_lang = LANGUAGE if LANGUAGE else None
            segments = await loop.run_in_executor(
                executor,
                align_segments,
                segments,
                audio_float32,
                align_lang,
                TORCH_DEVICE,
                ALIGN_MODEL_NAME,
                None  # cache_dir
            )
    except Exception as align_err:
        logger.warning(f"{session.log_prefix} Alignment skipped due to error: {align_err}")

    # Map diarization regions to session speakers
    diar_mapped = diar_regions
    if diarization_override is None and diar_regions and session.speaker_identifier and is_speaker_model_loaded():
        diar_mapped = await map_diarization_to_session_speakers(
            diar_regions,
            audio_float32,
            session,
            executor,
            loop
        )

    # Apply absolute time offset
    for region in diar_mapped:
        region["start"] += time_offset
        region["end"] += time_offset

    logger.info(
        f"{session.log_prefix} Chunk processed: segments={len(segments)}, "
        f"diar_regions_raw={len(diar_regions)}, diar_regions_mapped={len(diar_mapped)}, "
        f"duration={len(audio_float32)/SAMPLE_RATE:.2f}s"
    )

    return {
        "segments": segments,
        "diarization": diar_mapped
    }


def _pick_best_speaker_for_span(
    span_start: float,
    span_end: float,
    diar_segments: list
) -> str:
    """Return speaker label with the largest overlap for a time span."""
    best_label = None
    best_overlap = 0.0
    for region in diar_segments:
        overlap = min(span_end, region["end"]) - max(span_start, region["start"])
        if overlap > best_overlap and overlap > 0:
            best_overlap = overlap
            best_label = region.get("speaker")
    return best_label


def _slice_segment_with_diarization(segment: Dict, diar_segments: list) -> list:
    """
    Slice a transcript segment using diarization regions, returning
    per-speaker sub-segments. Falls back to dominant speaker if no words.
    """
    if not diar_segments:
        return [segment]

    words = segment.get("words") or []
    if not words:
        # No word alignment; use dominant diarization speaker over the segment
        dominant = _pick_best_speaker_for_span(segment["start"], segment["end"], diar_segments)
        if dominant:
            segment = {**segment, "speaker": dominant}
        return [segment]

    sub_segments = []
    current_words = []
    current_speaker = None

    for word in words:
        w_start = float(word.get("start", segment["start"]))
        w_end = float(word.get("end", segment["end"]))
        speaker = _pick_best_speaker_for_span(w_start, w_end, diar_segments)

        if current_speaker is None:
            current_speaker = speaker

        if speaker != current_speaker and current_words:
            text = "".join([w["word"] for w in current_words]).strip()
            sub_segments.append({
                "text": text,
                "start": float(current_words[0]["start"]),
                "end": float(current_words[-1]["end"]),
                "speaker": current_speaker
            })
            current_words = []

        current_speaker = speaker
        current_words.append(word)

    if current_words:
        text = "".join([w["word"] for w in current_words]).strip()
        sub_segments.append({
            "text": text,
            "start": float(current_words[0]["start"]),
            "end": float(current_words[-1]["end"]),
            "speaker": current_speaker
        })

    return sub_segments


def _int16_to_float32(audio_int16: np.ndarray) -> np.ndarray:
    if len(audio_int16) == 0:
        return np.array([], dtype=np.float32)
    return audio_int16.astype(np.float32) / 32768.0


async def _emit_pending_segments(
    session: Session,
    stable_time: float,
    allow_partial: bool,
    partial_debounce: float,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """
    Emit partial updates (throttled) and finalize segments once they are older
    than stable_time. Pending list is kept small by pruning finalized entries.
    """
    now = loop.time()
    can_send_partial = allow_partial and (now - session.last_partial_sent_at) >= partial_debounce

    keep: List[Dict[str, Any]] = []
    for entry in session.pending_segments:
        segment = entry["segment"]
        if not entry.get("sent_partial") and can_send_partial:
            await send_transcript(session, segment, is_final=False)
            entry["sent_partial"] = True
            session.last_partial_sent_at = now

        if not entry.get("sent_final") and segment["end"] <= stable_time:
            await send_transcript(session, segment, is_final=True)
            session.segments_processed += 1
            entry["sent_final"] = True

        if not entry.get("sent_final"):
            keep.append(entry)

    session.pending_segments = keep


async def _maybe_run_diarization(
    session: Session,
    executor,
    loop: asyncio.AbstractEventLoop,
    force: bool = False
) -> None:
    """Run diarization on the rolling buffer on a slower cadence to keep context."""
    if not (session.speaker_identifier and is_diarization_model_loaded()):
        return

    diar_len_seconds = session.diar_buffer.duration_seconds
    diar_buffer_end = session.diar_buffer_start_time + diar_len_seconds
    if diar_len_seconds < 0.5:
        return

    if not force and (diar_buffer_end - session.last_diar_run) < DIAR_HOP_SECONDS:
        return

    diar_audio_int16 = session.diar_buffer.get_all()
    if len(diar_audio_int16) == 0:
        return

    diar_audio_float32 = _int16_to_float32(diar_audio_int16)
    diar_regions = await loop.run_in_executor(executor, diarize_audio, diar_audio_float32)

    if diar_regions and session.speaker_identifier and is_speaker_model_loaded():
        diar_regions = await map_diarization_to_session_speakers(
            diar_regions,
            diar_audio_float32,
            session,
            executor,
            loop
        )

    # Offset to absolute timeline
    for region in diar_regions:
        region["start"] += session.diar_buffer_start_time
        region["end"] += session.diar_buffer_start_time

    session.diarization_cache = diar_regions
    session.last_diar_run = diar_buffer_end


async def _process_window(
    session: Session,
    window_audio_float32: np.ndarray,
    window_start: float,
    diar_regions: list,
    executor,
    loop: asyncio.AbstractEventLoop,
    stable_time: float,
    partial_debounce: float,
    allow_partial: bool,
) -> None:
    """
    Run ASR for a single streaming window and enqueue partial/final emissions.
    """
    if len(window_audio_float32) < int(0.25 * SAMPLE_RATE):
        return

    results = await run_transcribe_and_diarize(
        window_audio_float32,
        session,
        window_start,
        executor,
        loop,
        diarization_override=diar_regions
    )

    diar_for_window = diar_regions or results.get("diarization", [])
    segments = results.get("segments", [])
    logger.info(
        f"{session.log_prefix} Window processed: start={window_start:.2f}s "
        f"dur={len(window_audio_float32)/SAMPLE_RATE:.2f}s "
        f"segments={len(segments)}, diar_regions={len(diar_for_window)}"
    )

    for segment in segments:
        if not segment.get("text"):
            continue

        sub_segments = _slice_segment_with_diarization(segment, diar_for_window)

        for sub in sub_segments:
            if not sub.get("speaker") and session.speaker_identifier:
                seg_start = int((sub["start"] - window_start) * SAMPLE_RATE)
                seg_end = int((sub["end"] - window_start) * SAMPLE_RATE)
                seg_start = max(0, seg_start)
                seg_end = min(len(window_audio_float32), seg_end)
                segment_duration = (seg_end - seg_start) / SAMPLE_RATE

                if seg_end > seg_start and segment_duration >= 0.3:
                    segment_audio = window_audio_float32[seg_start:seg_end]
                    speaker = await identify_speaker_for_segment(
                        segment_audio, session, executor, loop
                    )
                    if speaker:
                        sub["speaker"] = speaker
                        session.last_known_speaker = speaker
                    elif session.speaker_identifier:
                        last_speaker = session.speaker_identifier.get_last_speaker_label()
                        if last_speaker:
                            sub["speaker"] = last_speaker
                            session.last_known_speaker = last_speaker

            session.pending_segments.append({
                "segment": sub,
                "sent_partial": False,
                "sent_final": False
            })

        # If speaker still missing but we have a recent known speaker, apply it
        if session.last_known_speaker:
            for entry in session.pending_segments:
                seg = entry["segment"]
                if not seg.get("speaker"):
                    seg["speaker"] = session.last_known_speaker

    await _emit_pending_segments(
        session,
        stable_time=stable_time,
        allow_partial=allow_partial,
        partial_debounce=partial_debounce,
        loop=loop
    )


async def _process_remaining_buffer(
    session: Session,
    executor,
    loop: asyncio.AbstractEventLoop,
    partial_debounce: float,
) -> None:
    """
    Flush whatever remains in the buffer when the client disconnects.
    """
    buffer_int16 = session.audio_buffer.get_all()
    if len(buffer_int16) == 0:
        return

    buffer_float32 = _int16_to_float32(buffer_int16)
    buffer_end_time = session.buffer_start_time + len(buffer_int16) / SAMPLE_RATE
    session.absolute_time_offset = buffer_end_time

    await _maybe_run_diarization(session, executor, loop, force=True)

    start_time = max(session.next_window_start, session.buffer_start_time)
    if buffer_end_time > start_time:
        start_idx = int((start_time - session.buffer_start_time) * SAMPLE_RATE)
        window_audio = buffer_float32[start_idx:]
        await _process_window(
            session,
            window_audio,
            start_time,
            session.diarization_cache,
            executor,
            loop,
            stable_time=buffer_end_time,
            partial_debounce=partial_debounce,
            allow_partial=False,
        )

    await _emit_pending_segments(
        session,
        stable_time=buffer_end_time,
        allow_partial=False,
        partial_debounce=partial_debounce,
        loop=loop
    )


async def process_transcription(session: Session):
    """
    Process audio chunks and transcribe them with speaker identification.
    
    Uses short overlapping windows for low-latency streaming while keeping
    a rolling diarization buffer for better speaker segmentation.
    """
    executor = get_executor()
    loop = asyncio.get_running_loop()
    window_size_samples = int(STREAM_WINDOW_SECONDS * SAMPLE_RATE)
    hop_seconds = STREAM_HOP_SECONDS
    partial_debounce = STREAM_PARTIAL_DEBOUNCE_MS / 1000.0
    send_partials = STREAM_ENABLE_PARTIALS

    while True:
        try:
            audio_chunk = await session.audio_queue.get()

            if audio_chunk is None:  # Sentinel value to stop
                await _process_remaining_buffer(
                    session,
                    executor,
                    loop,
                    partial_debounce,
                )
                break

            # Add chunk to rolling buffers and advance start times for trimmed samples
            trimmed_main = session.audio_buffer.append(audio_chunk)
            trimmed_diar = session.diar_buffer.append(audio_chunk)
            if trimmed_main:
                session.buffer_start_time += trimmed_main / SAMPLE_RATE
            if trimmed_diar:
                session.diar_buffer_start_time += trimmed_diar / SAMPLE_RATE

            buffer_int16 = session.audio_buffer.get_all()
            if len(buffer_int16) == 0:
                continue

            buffer_float32 = _int16_to_float32(buffer_int16)
            buffer_end_time = session.buffer_start_time + len(buffer_int16) / SAMPLE_RATE
            session.absolute_time_offset = buffer_end_time

            await _maybe_run_diarization(session, executor, loop)

            if session.next_window_start < session.buffer_start_time:
                session.next_window_start = session.buffer_start_time

            while session.next_window_start + STREAM_WINDOW_SECONDS <= buffer_end_time:
                start_idx = int((session.next_window_start - session.buffer_start_time) * SAMPLE_RATE)
                end_idx = start_idx + window_size_samples
                if end_idx > len(buffer_float32):
                    break

                window_audio = buffer_float32[start_idx:end_idx]
                stable_time = max(session.buffer_start_time, buffer_end_time - STREAM_FINAL_LAG_SECONDS)

                await _process_window(
                    session,
                    window_audio,
                    session.next_window_start,
                    session.diarization_cache,
                    executor,
                    loop,
                    stable_time=stable_time,
                    partial_debounce=partial_debounce,
                    allow_partial=send_partials,
                )

                session.next_window_start += hop_seconds

            stable_time = max(session.buffer_start_time, buffer_end_time - STREAM_FINAL_LAG_SECONDS)
            await _emit_pending_segments(
                session,
                stable_time=stable_time,
                allow_partial=send_partials,
                partial_debounce=partial_debounce,
                loop=loop
            )
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
    tag = "final" if is_final else "partial"
    logger.info(f"{session.log_prefix} Transcript({tag}): {segment['text']}{speaker_info} [{start_time:.2f}s - {end_time:.2f}s]")
    
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


def diag_summary(segments, diar_regions_raw, diar_regions_mapped, audio_len):
    return {
        "segments": len(segments),
        "diar_regions_raw": len(diar_regions_raw),
        "diar_regions_mapped": len(diar_regions_mapped),
        "duration_seconds": audio_len
    }
