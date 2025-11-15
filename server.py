# IMPORTANT: Import compatibility shims FIRST to patch libraries before pyannote.audio imports them
from app.utils.hf_hub_compat import *  # noqa: F403, F401
from app.utils.torchaudio_compat import *  # noqa: F403, F401

import asyncio
import signal
import os
import websockets

from app.config.settings import USE_VAD, USE_DIARIZATION, SERVER_HOST
from app.utils.logger import logger
from app.services.transcription import load_model, get_executor
from app.services.vad import load_vad_model
from app.services.diarization import load_diarization_model
from app.handlers.health_check import start_http_server
from app.handlers.websocket_handler import websocket_handler


async def main():
    """Main function to start the HTTP API server and handle shutdown."""
    # Load models in a separate thread to not block asyncio startup
    loop = asyncio.get_running_loop()
    executor = get_executor()
    
    # Load VAD model first if enabled (non-blocking - server will work without it)
    if USE_VAD:
        logger.info("VAD enabled - loading VAD model...")
        try:
            await loop.run_in_executor(executor, load_vad_model)
        except Exception as e:
            logger.warning(f"VAD loading failed: {e}. Continuing without VAD.")
    
    # Load Whisper model (required - server won't start without it)
    await loop.run_in_executor(executor, load_model)
    
    # Load diarization model if enabled (non-blocking - server will work without it)
    if USE_DIARIZATION:
        logger.info("Diarization enabled - loading diarization model...")
        try:
            await loop.run_in_executor(executor, load_diarization_model)
        except Exception as e:
            logger.warning(f"Diarization loading failed: {e}. Continuing without diarization.")

    # Start the HTTP API server (health check + transcription endpoint)
    http_site = await start_http_server()

    # Start the WebSocket server
    websocket_port = int(os.getenv("WEBSOCKET_PORT", "8765"))
    websocket_server = await websockets.serve(websocket_handler, SERVER_HOST, websocket_port)
    logger.info(f"WebSocket server started on ws://{SERVER_HOST}:{websocket_port}")

    # Set up signal handlers for graceful shutdown
    stop_event = asyncio.Event()
    loop.add_signal_handler(signal.SIGINT, stop_event.set)
    loop.add_signal_handler(signal.SIGTERM, stop_event.set)

    # Wait for shutdown signal
    await stop_event.wait()
    logger.info("Shutdown signal received. Closing server...")

    # Clean up resources
    websocket_server.close()
    await websocket_server.wait_closed()
    await http_site.stop()
    executor.shutdown(wait=True)
    logger.info("Server shut down gracefully.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped manually.")
