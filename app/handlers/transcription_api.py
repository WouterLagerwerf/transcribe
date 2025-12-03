"""HTTP API handler for transcription."""

import asyncio
import numpy as np
from aiohttp import web
from aiohttp.web_request import Request

from app.config.settings import SAMPLE_RATE, USE_DIARIZATION
from app.utils.logger import logger
from app.services.transcription import transcribe_synchronous, get_executor, is_server_ready
from app.services.speaker_identification import (
    SpeakerIdentifier, is_model_loaded as is_speaker_model_loaded, extract_embedding
)


async def transcribe_handler(request: Request):
    """
    HTTP POST handler for audio transcription.
    
    Accepts raw 16-bit PCM audio at 16kHz.
    Returns transcription with optional speaker identification.
    """
    if not is_server_ready():
        return web.json_response(
            {"error": "Server is not ready, model not loaded"},
            status=503
        )
    
    # Check content type
    content_type = request.headers.get('Content-Type', '')
    if not content_type.startswith('audio/'):
        return web.json_response(
            {"error": "Content-Type must be audio/* (e.g., audio/wav, audio/mpeg)"},
            status=400
        )
    
    try:
        # Read audio data
        audio_data_bytes = await request.read()
        
        if len(audio_data_bytes) == 0:
            return web.json_response(
                {"error": "Empty audio file"},
                status=400
            )
        
        # Convert bytes to numpy array (assuming 16-bit PCM, 16kHz, mono)
        audio_samples = np.frombuffer(audio_data_bytes, dtype=np.int16)
        
        # Convert to float32 for transcription
        audio_float32 = audio_samples.astype(np.float32) / 32768.0
        
        # Transcribe
        executor = get_executor()
        loop = asyncio.get_event_loop()
        segments = await loop.run_in_executor(
            executor, transcribe_synchronous, audio_float32, 0.0
        )
        
        if not segments:
            return web.json_response({
                "segments": [],
                "total_segments": 0,
                "full_text": "",
                "has_speakers": False
            })
        
        # Perform speaker identification if enabled
        has_speakers = False
        if USE_DIARIZATION and is_speaker_model_loaded():
            logger.info("Performing speaker identification...")
            
            # Create a speaker identifier for this request
            identifier = SpeakerIdentifier()
            
            # Identify speaker for each segment
            for segment in segments:
                # Extract segment audio
                seg_start = int(segment["start"] * SAMPLE_RATE)
                seg_end = int(segment["end"] * SAMPLE_RATE)
                seg_start = max(0, seg_start)
                seg_end = min(len(audio_float32), seg_end)
                
                if seg_end > seg_start:
                    segment_audio = audio_float32[seg_start:seg_end]
                    speaker_label, confidence = await loop.run_in_executor(
                        executor, identifier.identify_speaker, segment_audio
                    )
                    if speaker_label:
                        segment["speaker"] = speaker_label
                        has_speakers = True
            
            speaker_count = identifier.get_speaker_count()
            logger.info(f"Speaker identification completed: {speaker_count} speakers identified")
        
        # Return segments with timestamps (and speaker info if identification was performed)
        return web.json_response({
            "segments": segments,
            "total_segments": len(segments),
            "full_text": " ".join([s["text"] for s in segments]).strip(),
            "has_speakers": has_speakers
        })
        
    except ValueError as e:
        logger.error(f"Error processing audio: {e}", exc_info=True)
        return web.json_response(
            {"error": f"Invalid audio format: {str(e)}"},
            status=400
        )
    except Exception as e:
        logger.error(f"Error during transcription: {e}", exc_info=True)
        return web.json_response(
            {"error": "Internal server error during transcription"},
            status=500
        )
