"""WebSocket connection handling and audio processing with VAD support."""

import asyncio
import json
import numpy as np
import websockets

from app.config.settings import (
    SAMPLE_RATE, CHUNK_SIZE_SECONDS, USE_VAD,
    VAD_THRESHOLD, VAD_MIN_SILENCE_MS, VAD_SPEECH_PAD_MS, MAX_SEGMENT_SECONDS,
    VAD_MAX_SPEECH_MS, USE_DIARIZATION, REALTIME_INTERVAL_SECONDS
)
from app.utils.logger import logger
from app.services.transcription import transcribe_synchronous, get_executor, is_server_ready
from app.services.vad import load_vad_model, detect_speech_in_chunk, is_vad_loaded
from app.services.diarization import diarize_audio, assign_speakers_to_segments, is_diarization_loaded

# Global state for tracking connected clients
clients = set()


async def transcribe_and_send_realtime(websocket, audio_segment_float32, segment_start_time, executor, loop):
    """
    Transcribe audio segment and send partial (real-time) results.
    This runs in the background while speech is ongoing.
    """
    try:
        segments = await loop.run_in_executor(
            executor, transcribe_synchronous, audio_segment_float32, segment_start_time
        )
        
        # Send each segment as partial result (is_final: false)
        for segment in segments:
            if segment["text"]:
                response_data = {
                    "transcript": segment["text"],
                    "start": segment["start"],
                    "end": segment["end"],
                    "is_final": False  # Mark as partial/real-time result
                }
                
                response = json.dumps(response_data)
                try:
                    await websocket.send(response)
                    logger.debug(f"[{websocket.remote_address}] Real-time transcript: {segment['text']} [{segment['start']:.2f}s - {segment['end']:.2f}s]")
                except (websockets.exceptions.ConnectionClosed, websockets.exceptions.ConnectionClosedOK):
                    pass  # Connection closed, stop sending
    except Exception as e:
        logger.warning(f"[{websocket.remote_address}] Real-time transcription failed: {e}", exc_info=True)


def map_speakers_consistently(diarization_segments, speaker_map, speaker_history, segment_start_time, next_speaker_id):
    """
    Map pyannote speaker labels to consistent global speaker IDs across segments.
    
    Args:
        diarization_segments: List of diarization segments with 'speaker' labels
        speaker_map: Dict mapping pyannote labels to global IDs
        speaker_history: List of (global_speaker_id, end_time) tuples
        segment_start_time: Start time of current segment
        next_speaker_id: Next available global speaker ID
    
    Returns:
        (updated_segments, updated_speaker_map, updated_speaker_history, updated_next_speaker_id)
    """
    updated_segments = []
    updated_speaker_map = speaker_map.copy()
    updated_speaker_history = speaker_history.copy()
    updated_next_speaker_id = next_speaker_id
    
    # Clean old history (keep only speakers from last 10 seconds)
    current_time = segment_start_time
    updated_speaker_history = [(sid, end_time) for sid, end_time in updated_speaker_history 
                               if current_time - end_time < 10.0]
    
    # Get unique pyannote speakers in this segment
    pyannote_speakers = set(seg["speaker"] for seg in diarization_segments)
    
    # Map each pyannote speaker to a global ID
    for pyannote_speaker in pyannote_speakers:
        if pyannote_speaker not in updated_speaker_map:
            # New speaker detected - check if we can match to recent speaker
            best_match = None
            min_time_gap = float('inf')
            
            # Strategy 1: If there's only one speaker in history and one in new segment, likely same speaker
            if len(updated_speaker_history) == 1 and len(pyannote_speakers) == 1:
                best_match = updated_speaker_history[0][0]
                min_time_gap = segment_start_time - updated_speaker_history[0][1]
                logger.debug(f"Single speaker match: {pyannote_speaker} -> {best_match} (gap: {min_time_gap:.2f}s)")
            else:
                # Strategy 2: Find the most recent speaker that ended close to this segment start
                for global_id, end_time in updated_speaker_history:
                    time_gap = segment_start_time - end_time
                    # If speaker ended within 5 seconds before this segment, consider it a match
                    # Increased from 3s to 5s to handle longer pauses better
                    if 0 <= time_gap < 5.0 and time_gap < min_time_gap:
                        min_time_gap = time_gap
                        best_match = global_id
            
            if best_match is not None and min_time_gap < 5.0:
                # Map to existing speaker
                updated_speaker_map[pyannote_speaker] = best_match
                logger.debug(f"Mapped {pyannote_speaker} to existing speaker {best_match} (gap: {min_time_gap:.2f}s)")
            else:
                # New speaker - assign new global ID
                updated_speaker_map[pyannote_speaker] = updated_next_speaker_id
                updated_next_speaker_id += 1
                logger.debug(f"Mapped {pyannote_speaker} to new speaker {updated_speaker_map[pyannote_speaker]}")
    
    # Update segments with mapped speaker IDs
    for seg in diarization_segments:
        mapped_seg = seg.copy()
        mapped_seg["speaker"] = f"SPEAKER_{updated_speaker_map[seg['speaker']]:02d}"
        updated_segments.append(mapped_seg)
        
        # Update history with this speaker's end time
        global_id = updated_speaker_map[seg['speaker']]
        # Remove old entries for this speaker
        updated_speaker_history = [(sid, end_time) for sid, end_time in updated_speaker_history if sid != global_id]
        # Add new entry
        updated_speaker_history.append((global_id, seg["end"]))
    
    return updated_segments, updated_speaker_map, updated_speaker_history, updated_next_speaker_id


async def process_transcription_with_vad(websocket, audio_queue):
    """
    Manages VAD and transcription for a single client.
    Transcribes continuously during speech for real-time updates, and sends final results when speech ends.
    """
    full_audio_buffer = np.array([], dtype=np.int16)
    triggered = False
    speech_start_samples = 0
    silence_start_samples = 0
    speech_start_time_samples = 0  # Track when speech started
    last_transcription = ""
    absolute_time_offset = 0.0  # Track absolute time offset for timestamps
    last_realtime_transcription_time = 0.0  # Track when we last sent a real-time transcription
    last_realtime_transcription_samples = 0  # Track sample position of last real-time transcription
    executor = get_executor()
    loop = asyncio.get_running_loop()
    
    # Speaker tracking across segments
    speaker_map = {}  # Maps pyannote speaker labels (SPEAKER_00, etc.) to consistent global IDs
    speaker_history = []  # List of (speaker_id, end_time) tuples to track recent speakers
    next_speaker_id = 0  # Next available global speaker ID

    while True:
        try:
            audio_chunk = await audio_queue.get()
            if audio_chunk is None:  # Sentinel to stop
                # Process any remaining audio when connection closes
                if len(full_audio_buffer) > 0:
                    buffer_duration_seconds = len(full_audio_buffer) / SAMPLE_RATE
                    
                    if triggered:
                        # We were in the middle of speech - process from speech_start_samples to end
                        final_segment = full_audio_buffer[int(speech_start_samples):]
                    else:
                        # No active speech, but check if there's enough audio to transcribe
                        final_segment = full_audio_buffer
                    
                    # Limit final segment to max size
                    max_segment_samples = int(SAMPLE_RATE * MAX_SEGMENT_SECONDS)
                    if len(final_segment) > max_segment_samples:
                        logger.warning(f"[{websocket.remote_address}] Buffer exceeded max size ({buffer_duration_seconds:.2f}s), truncating to {MAX_SEGMENT_SECONDS}s...")
                        final_segment = final_segment[:max_segment_samples]
                    
                    if len(final_segment) > SAMPLE_RATE * 0.5:  # At least 500ms of audio
                        speech_segment_float32 = final_segment.astype(np.float32) / 32768.0
                        # Calculate time offset for this segment
                        segment_start_time = absolute_time_offset + (speech_start_samples / SAMPLE_RATE if triggered else 0)
                        segments = await loop.run_in_executor(
                            executor, transcribe_synchronous, speech_segment_float32, segment_start_time
                        )
                        
                        # Perform speaker diarization if enabled
                        if USE_DIARIZATION and is_diarization_loaded() and segments:
                            try:
                                diarization_segments = await loop.run_in_executor(
                                    executor, diarize_audio, speech_segment_float32
                                )
                                if diarization_segments:
                                    # Adjust diarization timestamps to match absolute time
                                    adjusted_diarization = []
                                    for diar_seg in diarization_segments:
                                        adjusted_diarization.append({
                                            "start": diar_seg["start"] + segment_start_time,
                                            "end": diar_seg["end"] + segment_start_time,
                                            "speaker": diar_seg["speaker"]
                                        })
                                    
                                    # Map speakers consistently across segments
                                    mapped_diarization, speaker_map, speaker_history, next_speaker_id = map_speakers_consistently(
                                        adjusted_diarization, speaker_map, speaker_history, segment_start_time, next_speaker_id
                                    )
                                    
                                    segments = assign_speakers_to_segments(segments, mapped_diarization)
                                    logger.info(f"[{websocket.remote_address}] Final diarization completed: {len(mapped_diarization)} speaker segments")
                            except Exception as e:
                                logger.warning(f"[{websocket.remote_address}] Final diarization failed: {e}", exc_info=True)
                        
                        # Send each segment with timestamps (and speaker if available)
                        for segment in segments:
                            if segment["text"]:
                                speaker_info = f" [{segment.get('speaker', 'UNKNOWN')}]" if segment.get('speaker') else ""
                                logger.info(f"[{websocket.remote_address}] Final transcript: {segment['text']}{speaker_info} [{segment['start']:.2f}s - {segment['end']:.2f}s]")
                                response_data = {
                                    "transcript": segment["text"],
                                    "start": segment["start"],
                                    "end": segment["end"],
                                    "is_final": True
                                }
                                # Add speaker if available
                                if segment.get("speaker"):
                                    response_data["speaker"] = segment["speaker"]
                                
                                response = json.dumps(response_data)
                                try:
                                    await websocket.send(response)
                                except (websockets.exceptions.ConnectionClosed, websockets.exceptions.ConnectionClosedOK):
                                    pass  # Connection already closed
                break

            # Add chunk to buffer
            full_audio_buffer = np.concatenate([full_audio_buffer, audio_chunk])
            buffer_duration_ms = (len(full_audio_buffer) / SAMPLE_RATE) * 1000
            buffer_duration_seconds = buffer_duration_ms / 1000.0
            
            # Safety check: force transcription if buffer exceeds max segment size (for memory safety)
            force_transcription = False
            if triggered:
                speech_duration_samples = len(full_audio_buffer) - speech_start_time_samples
                speech_duration_ms = (speech_duration_samples / SAMPLE_RATE) * 1000
                speech_duration_seconds = speech_duration_ms / 1000.0
                
                if buffer_duration_seconds > MAX_SEGMENT_SECONDS:
                    if silence_start_samples == 0:  # Only log once
                        logger.warning(f"[{websocket.remote_address}] Buffer exceeded max size ({buffer_duration_seconds:.1f}s > {MAX_SEGMENT_SECONDS}s), forcing transcription...")
                    force_transcription = True
                    # Set up silence tracking to trigger transcription
                    if silence_start_samples == 0:
                        silence_start_samples = len(full_audio_buffer) - len(audio_chunk)
            
            # Only run VAD if we have enough samples (at least 512 samples = 32ms at 16kHz)
            if len(full_audio_buffer) >= 512:
                # Run VAD detection on the latest audio (detect_speech_in_chunk handles chunking)
                speech_prob = await loop.run_in_executor(executor, detect_speech_in_chunk, full_audio_buffer)
                
                # Initialize silence_duration_ms for force_transcription case
                silence_duration_ms = 0
                if triggered and silence_start_samples > 0:
                    silence_duration_samples = len(full_audio_buffer) - silence_start_samples
                    silence_duration_ms = (silence_duration_samples / SAMPLE_RATE) * 1000
                
                if speech_prob > VAD_THRESHOLD:
                    # Speech detected
                    if force_transcription:
                        # Force transcription is set - will trigger transcription below
                        pass
                    elif silence_start_samples > 0:
                        # We were in silence, but now speech is detected again - reset silence tracking
                        silence_start_samples = 0
                    
                    if not triggered:
                        # Speech just started - mark the start position
                        triggered = True
                        speech_pad_samples = int(SAMPLE_RATE * VAD_SPEECH_PAD_MS / 1000)
                        # When speech starts, we want to include a bit of audio before detection
                        # If buffer is small (< 1 second), likely just reset - start from beginning
                        # Otherwise, go back by pad amount from current position
                        buffer_samples = len(full_audio_buffer)
                        if buffer_samples < SAMPLE_RATE:  # Less than 1 second
                            # Buffer is small, likely just reset - start from beginning
                            speech_start_samples = 0
                        else:
                            # Buffer has enough audio - go back by pad amount from current position
                            current_pos = buffer_samples - len(audio_chunk)
                            speech_start_samples = max(0, current_pos - speech_pad_samples)
                        speech_start_time_samples = len(full_audio_buffer) - len(audio_chunk)  # Track when speech started
                        last_realtime_transcription_time = 0.0  # Reset real-time transcription timer
                        last_realtime_transcription_samples = speech_start_samples  # Reset real-time transcription position
                    
                    # Real-time transcription: transcribe periodically during continuous speech
                    if triggered and not force_transcription:
                        current_time = absolute_time_offset + (len(full_audio_buffer) / SAMPLE_RATE)
                        time_since_last_realtime = current_time - last_realtime_transcription_time
                        
                        # Check if enough time has passed for real-time transcription
                        if time_since_last_realtime >= REALTIME_INTERVAL_SECONDS:
                            # Get audio segment from last transcription point to now
                            realtime_segment_start = max(speech_start_samples, last_realtime_transcription_samples)
                            realtime_segment_end = len(full_audio_buffer)
                            
                            # Only transcribe if we have at least 0.5 seconds of new audio
                            realtime_segment_duration = (realtime_segment_end - realtime_segment_start) / SAMPLE_RATE
                            if realtime_segment_duration >= 0.5:  # At least 500ms of new audio
                                realtime_segment = full_audio_buffer[int(realtime_segment_start):int(realtime_segment_end)]
                                realtime_segment_float32 = realtime_segment.astype(np.float32) / 32768.0
                                realtime_segment_start_time = absolute_time_offset + (realtime_segment_start / SAMPLE_RATE)
                                
                                # Transcribe in background (don't await - allow processing to continue)
                                asyncio.create_task(transcribe_and_send_realtime(
                                    websocket, realtime_segment_float32, realtime_segment_start_time, executor, loop
                                ))
                                
                                # Update tracking
                                last_realtime_transcription_time = current_time
                                last_realtime_transcription_samples = realtime_segment_end
                else:
                    # Silence detected
                    if triggered:
                        if silence_start_samples == 0:
                            silence_start_samples = len(full_audio_buffer) - len(audio_chunk)
                        
                        # Calculate silence duration
                        silence_duration_samples = len(full_audio_buffer) - silence_start_samples
                        silence_duration_ms = (silence_duration_samples / SAMPLE_RATE) * 1000
                
                # Check if we should transcribe (either silence detected OR force transcription)
                if triggered and (force_transcription or (speech_prob <= VAD_THRESHOLD and silence_duration_ms >= VAD_MIN_SILENCE_MS)):
                    # Extract speech segment with padding
                    speech_pad_samples = int(SAMPLE_RATE * VAD_SPEECH_PAD_MS / 1000)
                    speech_end_samples = min(len(full_audio_buffer), silence_start_samples + speech_pad_samples)
                    
                    # Limit segment to max size if needed
                    max_segment_samples = int(SAMPLE_RATE * MAX_SEGMENT_SECONDS)
                    if speech_end_samples - speech_start_samples > max_segment_samples:
                        speech_end_samples = speech_start_samples + max_segment_samples
                        logger.warning(f"[{websocket.remote_address}] Buffer segment truncated to {MAX_SEGMENT_SECONDS}s to prevent memory issues.")
                    
                    speech_segment = full_audio_buffer[int(speech_start_samples):int(speech_end_samples)]
                    
                    if len(speech_segment) > 0:
                        speech_segment_float32 = speech_segment.astype(np.float32) / 32768.0
                        
                        # Calculate time offset for this segment (absolute time from start of stream)
                        segment_start_time = absolute_time_offset + (speech_start_samples / SAMPLE_RATE)
                        
                        # Transcribe
                        segments = await loop.run_in_executor(
                            executor, transcribe_synchronous, speech_segment_float32, segment_start_time
                        )
                        
                        # Perform speaker diarization if enabled
                        if USE_DIARIZATION and is_diarization_loaded() and segments:
                            try:
                                diarization_segments = await loop.run_in_executor(
                                    executor, diarize_audio, speech_segment_float32
                                )
                                if diarization_segments:
                                    # Adjust diarization timestamps to match absolute time
                                    adjusted_diarization = []
                                    for diar_seg in diarization_segments:
                                        adjusted_diarization.append({
                                            "start": diar_seg["start"] + segment_start_time,
                                            "end": diar_seg["end"] + segment_start_time,
                                            "speaker": diar_seg["speaker"]
                                        })
                                    
                                    # Map speakers consistently across segments
                                    mapped_diarization, speaker_map, speaker_history, next_speaker_id = map_speakers_consistently(
                                        adjusted_diarization, speaker_map, speaker_history, segment_start_time, next_speaker_id
                                    )
                                    
                                    segments = assign_speakers_to_segments(segments, mapped_diarization)
                                    logger.info(f"[{websocket.remote_address}] Diarization completed: {len(mapped_diarization)} speaker segments, active speakers: {len(set(s['speaker'] for s in mapped_diarization))}")
                            except Exception as e:
                                logger.warning(f"[{websocket.remote_address}] Diarization failed: {e}", exc_info=True)
                        
                        # Send each segment with timestamps (and speaker if available)
                        for segment in segments:
                            if segment["text"]:
                                speaker_info = f" [{segment.get('speaker', 'UNKNOWN')}]" if segment.get('speaker') else ""
                                logger.info(f"[{websocket.remote_address}] Transcript: {segment['text']}{speaker_info} [{segment['start']:.2f}s - {segment['end']:.2f}s]")
                                response_data = {
                                    "transcript": segment["text"],
                                    "start": segment["start"],
                                    "end": segment["end"],
                                    "is_final": True
                                }
                                # Add speaker if available
                                if segment.get("speaker"):
                                    response_data["speaker"] = segment["speaker"]
                                
                                response = json.dumps(response_data)
                                try:
                                    await websocket.send(response)
                                except (websockets.exceptions.ConnectionClosed, websockets.exceptions.ConnectionClosedOK):
                                    break
                        
                        # Update absolute time offset - we've processed up to speech_end_samples
                        # The remaining buffer starts at speech_end_samples, so update offset accordingly
                        absolute_time_offset += speech_end_samples / SAMPLE_RATE
                    
                    # Reset for next utterance - keep unprocessed audio in buffer
                    triggered = False
                    silence_start_samples = 0
                    speech_start_time_samples = 0
                    remaining_buffer = full_audio_buffer[int(speech_end_samples):]
                    full_audio_buffer = remaining_buffer
                    speech_start_samples = 0
                    # Reset real-time transcription tracking for next speech segment
                    last_realtime_transcription_time = 0.0
                    last_realtime_transcription_samples = 0

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"[{websocket.remote_address}] Error in VAD processor: {e}", exc_info=True)
            break


async def process_transcription_time_based(websocket, audio_queue):
    """
    Manages transcription using time-based chunking (fallback when VAD disabled).
    Processes audio in fixed time intervals.
    """
    current_audio_segment = np.array([], dtype=np.int16)
    last_transcription = ""
    absolute_time_offset = 0.0  # Track absolute time offset for timestamps
    executor = get_executor()
    loop = asyncio.get_running_loop()
    
    # Speaker tracking across segments
    speaker_map = {}  # Maps pyannote speaker labels (SPEAKER_00, etc.) to consistent global IDs
    speaker_history = []  # List of (speaker_id, end_time) tuples to track recent speakers
    next_speaker_id = 0  # Next available global speaker ID

    while True:
        try:
            audio_chunk = await audio_queue.get()
            if audio_chunk is None:  # Sentinel value to stop
                # Process any remaining audio
                if len(current_audio_segment) > 0:
                    audio_float32 = current_audio_segment.astype(np.float32) / 32768.0
                    segments = await loop.run_in_executor(
                        executor, transcribe_synchronous, audio_float32, absolute_time_offset
                    )
                    
                    # Perform speaker diarization if enabled
                    if USE_DIARIZATION and is_diarization_loaded() and segments:
                        try:
                            diarization_segments = await loop.run_in_executor(
                                executor, diarize_audio, audio_float32
                            )
                            if diarization_segments:
                                # Adjust diarization timestamps to match absolute time
                                adjusted_diarization = []
                                for diar_seg in diarization_segments:
                                    adjusted_diarization.append({
                                        "start": diar_seg["start"] + absolute_time_offset,
                                        "end": diar_seg["end"] + absolute_time_offset,
                                        "speaker": diar_seg["speaker"]
                                    })
                                
                                # Map speakers consistently across segments
                                mapped_diarization, speaker_map, speaker_history, next_speaker_id = map_speakers_consistently(
                                    adjusted_diarization, speaker_map, speaker_history, absolute_time_offset, next_speaker_id
                                )
                                
                                segments = assign_speakers_to_segments(segments, mapped_diarization)
                        except Exception as e:
                            logger.warning(f"[{websocket.remote_address}] Final diarization failed: {e}", exc_info=True)
                    
                    # Send each segment with timestamps (and speaker if available)
                    for segment in segments:
                        if segment["text"]:
                            speaker_info = f" [{segment.get('speaker', 'UNKNOWN')}]" if segment.get('speaker') else ""
                            logger.info(f"[{websocket.remote_address}] Final transcript: {segment['text']}{speaker_info} [{segment['start']:.2f}s - {segment['end']:.2f}s]")
                            response_data = {
                                "transcript": segment["text"],
                                "start": segment["start"],
                                "end": segment["end"],
                                "is_final": True
                            }
                            if segment.get("speaker"):
                                response_data["speaker"] = segment["speaker"]
                            response = json.dumps(response_data)
                            try:
                                await websocket.send(response)
                            except (websockets.exceptions.ConnectionClosed, websockets.exceptions.ConnectionClosedOK):
                                pass
                break

            current_audio_segment = np.concatenate((current_audio_segment, audio_chunk))
            segment_duration_seconds = len(current_audio_segment) / SAMPLE_RATE
            
            if segment_duration_seconds >= CHUNK_SIZE_SECONDS:
                segment_to_transcribe = current_audio_segment
                current_audio_segment = np.array([], dtype=np.int16)
                audio_float32 = segment_to_transcribe.astype(np.float32) / 32768.0

                segments = await loop.run_in_executor(
                    executor, transcribe_synchronous, audio_float32, absolute_time_offset
                )

                # Perform speaker diarization if enabled
                if USE_DIARIZATION and is_diarization_loaded() and segments:
                    try:
                        diarization_segments = await loop.run_in_executor(
                            executor, diarize_audio, audio_float32
                        )
                        if diarization_segments:
                            # Adjust diarization timestamps to match absolute time
                            adjusted_diarization = []
                            for diar_seg in diarization_segments:
                                adjusted_diarization.append({
                                    "start": diar_seg["start"] + absolute_time_offset,
                                    "end": diar_seg["end"] + absolute_time_offset,
                                    "speaker": diar_seg["speaker"]
                                })
                            
                            # Map speakers consistently across segments
                            mapped_diarization, speaker_map, speaker_history, next_speaker_id = map_speakers_consistently(
                                adjusted_diarization, speaker_map, speaker_history, absolute_time_offset, next_speaker_id
                            )
                            
                            segments = assign_speakers_to_segments(segments, mapped_diarization)
                    except Exception as e:
                        logger.warning(f"[{websocket.remote_address}] Diarization failed: {e}", exc_info=True)

                # Send each segment with timestamps (and speaker if available)
                for segment in segments:
                    if segment["text"]:
                        speaker_info = f" [{segment.get('speaker', 'UNKNOWN')}]" if segment.get('speaker') else ""
                        logger.info(f"[{websocket.remote_address}] Transcript: {segment['text']}{speaker_info} [{segment['start']:.2f}s - {segment['end']:.2f}s]")
                        response_data = {
                            "transcript": segment["text"],
                            "start": segment["start"],
                            "end": segment["end"],
                            "is_final": True
                        }
                        if segment.get("speaker"):
                            response_data["speaker"] = segment["speaker"]
                        response = json.dumps(response_data)
                        try:
                            await websocket.send(response)
                        except (websockets.exceptions.ConnectionClosed, websockets.exceptions.ConnectionClosedOK):
                            break
                
                # Update absolute time offset for next segment
                absolute_time_offset += segment_duration_seconds

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"[{websocket.remote_address}] Error in processor: {e}", exc_info=True)
            break


async def process_transcription_for_client(websocket, audio_queue):
    """
    Routes to VAD-based or time-based transcription based on configuration.
    """
    if USE_VAD and is_vad_loaded():
        await process_transcription_with_vad(websocket, audio_queue)
    else:
        await process_transcription_time_based(websocket, audio_queue)


async def websocket_handler(websocket):
    """Handles incoming WebSocket connections."""
    if not is_server_ready():
        await websocket.close(code=1013, reason="Server is not ready, model not loaded.")
        return

    clients.add(websocket)

    audio_queue = asyncio.Queue()
    processor_task = asyncio.create_task(process_transcription_for_client(websocket, audio_queue))

    try:
        async for message in websocket:
            if isinstance(message, bytes):
                # Put raw audio bytes into the queue for the processor task
                audio_chunk = np.frombuffer(message, dtype=np.int16)
                await audio_queue.put(audio_chunk)

    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception as e:
        logger.error(f"[{websocket.remote_address}] Connection error: {e}", exc_info=True)
    finally:
        # Signal the processor task to stop and wait for it to finish
        await audio_queue.put(None)
        await processor_task

        clients.remove(websocket)


def get_clients():
    """Returns the set of connected clients."""
    return clients

