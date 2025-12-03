# IMPORTANT: Import compatibility shims FIRST to patch libraries before pyannote.audio imports them
from app.utils.hf_hub_compat import *  # noqa: F403, F401
from app.utils.torchaudio_compat import *  # noqa: F403, F401

import asyncio
import signal
import os
import websockets

from app.config.settings import USE_DIARIZATION, SERVER_HOST
from app.utils.logger import logger
from app.services.transcription import load_model, get_executor
from app.services.speaker_identification import load_speaker_model
from app.handlers.health_check import start_http_server
from app.handlers.websocket_handler import websocket_handler


async def main():
    """Main function to start the HTTP API server and handle shutdown."""
    import sys
    from app.config.settings import DEVICE, COMPUTE_TYPE
    
    # Print startup banner with device info
    print("", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print("🚀 STARTING TRANSCRIPTION SERVER", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print(f"📱 Primary Device: {DEVICE.upper()}", file=sys.stderr)
    print(f"⚙️  Compute Type: {COMPUTE_TYPE}", file=sys.stderr)
    print(f"🎤 Speaker ID: {'Enabled' if USE_DIARIZATION else 'Disabled'}", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print("", file=sys.stderr)
    
    # Load models in a separate thread to not block asyncio startup
    loop = asyncio.get_running_loop()
    executor = get_executor()
    
    # Load Whisper model (required - server won't start without it)
    # Note: VAD is handled by faster-whisper's built-in Silero VAD
    await loop.run_in_executor(executor, load_model)
    
    # Load speaker embedding model if diarization is enabled
    if USE_DIARIZATION:
        logger.info("Speaker identification enabled - loading embedding model...")
        try:
            await loop.run_in_executor(executor, load_speaker_model)
        except Exception as e:
            logger.warning(f"Speaker model loading failed: {e}. Continuing without speaker identification.")
    
    # Print summary after all models loaded
    print("", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print("✅ ALL MODELS LOADED - SERVER READY", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print("", file=sys.stderr)

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
