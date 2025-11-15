"""HTTP API handler for transcription."""

import numpy as np
from aiohttp import web
from aiohttp.web_request import Request

from app.config.settings import SAMPLE_RATE, USE_DIARIZATION
from app.utils.logger import logger
from app.services.transcription import transcribe_synchronous, get_executor, is_server_ready
from app.services.vad import is_vad_loaded
from app.services.diarization import (
    load_diarization_model, diarize_audio, assign_speakers_to_segments, is_diarization_loaded
)
from app.config.settings import USE_VAD


async def transcribe_handler(request: Request):
    """HTTP POST handler for audio transcription."""
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
        # This matches what WebSocket clients send
        audio_samples = np.frombuffer(audio_data_bytes, dtype=np.int16)
        
        # Convert to float32 for transcription
        audio_float32 = audio_samples.astype(np.float32) / 32768.0
        
        # Transcribe
        import asyncio
        executor = get_executor()
        loop = asyncio.get_event_loop()
        segments = await loop.run_in_executor(
            executor, transcribe_synchronous, audio_float32, 0.0
        )
        
        if not segments:
            return web.json_response(
                {"error": "No transcription generated"},
                status=500
            )
        
        # Perform speaker diarization if enabled
        if USE_DIARIZATION:
            if is_diarization_loaded():
                logger.info("Performing speaker diarization...")
                diarization_segments = await loop.run_in_executor(
                    executor, diarize_audio, audio_float32
                )
                logger.info(f"Diarization returned {len(diarization_segments)} speaker segments")
                if diarization_segments:
                    logger.info(f"First few diarization segments: {diarization_segments[:3]}")
                    segments = assign_speakers_to_segments(segments, diarization_segments)
                    logger.info(f"Speaker diarization completed: {len(diarization_segments)} speaker segments found, {len(segments)} transcription segments assigned")
                    logger.info(f"Sample segment with speaker: {segments[0] if segments else 'None'}")
                else:
                    logger.warning("Diarization returned no segments")
            else:
                logger.warning("Diarization is enabled but model is not loaded. Check server logs for diarization model loading errors.")
        
        # Return segments with timestamps (and speaker info if diarization was performed)
        return web.json_response({
            "segments": segments,
            "total_segments": len(segments),
            "full_text": " ".join([s["text"] for s in segments]).strip(),
            "has_speakers": USE_DIARIZATION and is_diarization_loaded() and any("speaker" in s for s in segments)
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

