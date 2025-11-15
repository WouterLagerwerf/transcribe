import asyncio
import websockets
import json
import numpy as np
import logging
import os
import signal
from concurrent.futures import ThreadPoolExecutor
from aiohttp import web
import torch
from faster_whisper import WhisperModel

# --- Configuration (from Environment Variables) ---
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", 8765))
HEALTH_CHECK_PORT = int(os.getenv("HEALTH_CHECK_PORT", 8080))

# faster-whisper model configuration
MODEL_SIZE = os.getenv("MODEL_SIZE", "base")  # "tiny", "base", "small", "medium", "large"
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")  # "int8", "float16", "float32"
CPU_THREADS = int(os.getenv("CPU_THREADS", 4))
LANGUAGE = os.getenv("LANGUAGE", None)  # None for auto-detection

# VAD configuration
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", 0.5))
VAD_MIN_SILENCE_MS = int(os.getenv("VAD_MIN_SILENCE_MS", 500))  # Pauses to detect end of utterance
VAD_SPEECH_PAD_MS = int(os.getenv("VAD_SPEECH_PAD_MS", 100))  # Padding around speech

SAMPLE_RATE = 16000

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global State ---
whisper_model = None
vad_model = None
executor = ThreadPoolExecutor(max_workers=CPU_THREADS)
server_ready = False
clients = set()


# --- Core Logic ---

def load_models():
    """Loads the VAD and Whisper models into memory."""
    global whisper_model, vad_model, server_ready
    logger.info("Loading Silero VAD model...")
    try:
        vad_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )
        vad_model.eval()  # Set to evaluation mode
        logger.info("Silero VAD model loaded.")
    except Exception as e:
        logger.error(f"Failed to load Silero VAD model: {e}", exc_info=True)
        return

    logger.info(f"Loading faster-whisper model '{MODEL_SIZE}' with compute_type '{COMPUTE_TYPE}'...")
    try:
        whisper_model = WhisperModel(
            MODEL_SIZE,
            device="cpu",
            compute_type=COMPUTE_TYPE,
            cpu_threads=CPU_THREADS
        )
        logger.info(f"faster-whisper model '{MODEL_SIZE}' loaded.")
        server_ready = True
    except Exception as e:
        logger.error(f"Failed to load faster-whisper model: {e}", exc_info=True)


def run_vad(audio_float32: np.ndarray) -> float:
    """Run VAD model synchronously (should be called from executor)."""
    with torch.no_grad():
        speech_prob = vad_model(torch.from_numpy(audio_float32), SAMPLE_RATE).item()
    return speech_prob


def transcribe_synchronous(audio_data_float32: np.ndarray) -> str:
    """Synchronous wrapper for faster-whisper transcription."""
    if whisper_model is None:
        logger.error("Transcription called but model is not loaded.")
        return ""
    try:
        segments, _ = whisper_model.transcribe(
            audio_data_float32,
            beam_size=5,
            language=LANGUAGE if LANGUAGE else None
        )
        # Concatenate all segments to form the full transcription
        return " ".join([segment.text for segment in segments]).strip()
    except Exception as e:
        logger.error(f"Error during transcription: {e}", exc_info=True)
        return ""


async def process_transcription_for_client(websocket, audio_queue):
    """Manages VAD and transcription for a single client."""
    logger.info(f"[{websocket.remote_address}] VAD processor started.")
    full_audio_buffer = np.array([], dtype=np.int16)
    triggered = False
    speech_start_samples = 0
    silence_start_samples = 0
    last_transcription = ""

    loop = asyncio.get_running_loop()

    while True:
        try:
            audio_chunk_bytes = await audio_queue.get()
            if audio_chunk_bytes is None:  # Sentinel to stop
                # Process any remaining audio when connection closes
                if triggered and len(full_audio_buffer) > 0:
                    logger.info(f"[{websocket.remote_address}] Processing final audio segment...")
                    speech_segment_float32 = full_audio_buffer.astype(np.float32) / 32768.0
                    transcribed_text = await loop.run_in_executor(
                        executor, transcribe_synchronous, speech_segment_float32
                    )
                    if transcribed_text and transcribed_text != last_transcription:
                        logger.info(f"[{websocket.remote_address}] Final transcript: {transcribed_text}")
                        response = json.dumps({"transcript": transcribed_text, "is_final": True})
                        try:
                            await websocket.send(response)
                        except (websockets.exceptions.ConnectionClosed, websockets.exceptions.ConnectionClosedOK):
                            pass  # Connection already closed
                break

            # Convert chunk to numpy array
            audio_chunk_int16 = np.frombuffer(audio_chunk_bytes, dtype=np.int16)
            full_audio_buffer = np.concatenate([full_audio_buffer, audio_chunk_int16])
            
            # Convert to float32 for VAD
            audio_chunk_float32 = audio_chunk_int16.astype(np.float32) / 32768.0

            # --- VAD Logic (run in executor to avoid blocking) ---
            speech_prob = await loop.run_in_executor(executor, run_vad, audio_chunk_float32)

            if speech_prob > VAD_THRESHOLD:
                # Speech detected
                silence_start_samples = 0  # Reset silence timer
                if not triggered:
                    # Speech just started
                    triggered = True
                    # Add padding before speech start
                    speech_pad_samples = int(SAMPLE_RATE * VAD_SPEECH_PAD_MS / 1000)
                    speech_start_samples = max(0, len(full_audio_buffer) - len(audio_chunk_int16) - speech_pad_samples)
                    logger.info(f"[{websocket.remote_address}] Speech started.")
            else:
                # Silence detected
                if triggered:
                    # We were in a speech segment
                    if silence_start_samples == 0:
                        # Silence just started
                        silence_start_samples = len(full_audio_buffer) - len(audio_chunk_int16)
                    
                    # Calculate silence duration in samples
                    silence_duration_samples = len(full_audio_buffer) - silence_start_samples
                    silence_duration_ms = (silence_duration_samples / SAMPLE_RATE) * 1000
                    
                    if silence_duration_ms >= VAD_MIN_SILENCE_MS:
                        # Enough silence detected - end of utterance
                        logger.info(f"[{websocket.remote_address}] Speech ended (silence: {silence_duration_ms:.0f}ms).")
                        
                        # Extract the speech segment with padding
                        speech_pad_samples = int(SAMPLE_RATE * VAD_SPEECH_PAD_MS / 1000)
                        speech_end_samples = silence_start_samples + speech_pad_samples
                        speech_segment = full_audio_buffer[int(speech_start_samples):int(speech_end_samples)]
                        
                        if len(speech_segment) > 0:
                            # Convert to float32 for transcription
                            speech_segment_float32 = speech_segment.astype(np.float32) / 32768.0

                            # Run transcription in thread pool
                            transcribed_text = await loop.run_in_executor(
                                executor, transcribe_synchronous, speech_segment_float32
                            )

                            if transcribed_text and transcribed_text != last_transcription:
                                logger.info(f"[{websocket.remote_address}] Transcript: {transcribed_text}")
                                response = json.dumps({"transcript": transcribed_text, "is_final": True})
                                try:
                                    await websocket.send(response)
                                    last_transcription = transcribed_text
                                except (websockets.exceptions.ConnectionClosed, websockets.exceptions.ConnectionClosedOK) as e:
                                    logger.info(f"[{websocket.remote_address}] Connection closed while sending transcript: {e.code}")
                                    break

                        # Reset state for next utterance
                        triggered = False
                        silence_start_samples = 0
                        # Keep remaining buffer (after speech end)
                        full_audio_buffer = full_audio_buffer[int(speech_end_samples):]
                        speech_start_samples = 0

        except asyncio.CancelledError:
            logger.info(f"[{websocket.remote_address}] VAD processor cancelled.")
            break
        except Exception as e:
            logger.error(f"[{websocket.remote_address}] Error in processor: {e}", exc_info=True)
            break
    
    logger.info(f"[{websocket.remote_address}] VAD processor stopped.")


async def websocket_handler(websocket):
    """Handles incoming WebSocket connections."""
    if not server_ready:
        await websocket.close(code=1013, reason="Server is not ready, models not loaded.")
        return

    clients.add(websocket)
    logger.info(f"Client connected: {websocket.remote_address} (Total: {len(clients)})")
    audio_queue = asyncio.Queue()
    processor_task = asyncio.create_task(process_transcription_for_client(websocket, audio_queue))

    try:
        async for message in websocket:
            if isinstance(message, bytes):
                await audio_queue.put(message)
            else:
                logger.warning(f"[{websocket.remote_address}] Received non-binary message, ignoring.")
    except websockets.exceptions.ConnectionClosed as e:
        logger.info(f"Client disconnected: {e.reason} (Code: {e.code})")
    except Exception as e:
        logger.error(f"[{websocket.remote_address}] Connection error: {e}", exc_info=True)
    finally:
        await audio_queue.put(None)  # Signal processor to stop
        await processor_task
        clients.remove(websocket)
        logger.info(f"Cleaned up for client {websocket.remote_address} (Total: {len(clients)})")


async def health_check_handler(request):
    """Health check endpoint."""
    return web.json_response({
        "status": "ok" if server_ready else "unavailable",
        "model": MODEL_SIZE,
        "clients": len(clients)
    })


async def start_health_check_server():
    """Start the HTTP health check server."""
    app = web.Application()
    app.router.add_get("/health", health_check_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, SERVER_HOST, HEALTH_CHECK_PORT)
    await site.start()
    logger.info(f"Health check server running on http://{SERVER_HOST}:{HEALTH_CHECK_PORT}")
    return site


async def main():
    """Main function to start the server and handle shutdown."""
    # Load models in executor to not block asyncio startup
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(executor, load_models)
    
    # Start the health check server
    health_check_site = await start_health_check_server()
    
    # Set up signal handlers for graceful shutdown
    stop_event = asyncio.Event()
    loop.add_signal_handler(signal.SIGINT, stop_event.set)
    loop.add_signal_handler(signal.SIGTERM, stop_event.set)
    
    # Start the WebSocket server
    async with websockets.serve(websocket_handler, SERVER_HOST, SERVER_PORT) as server:
        logger.info(f"WebSocket server started on ws://{SERVER_HOST}:{SERVER_PORT}")
        await stop_event.wait()
        logger.info("Shutdown signal received. Closing server...")
        # Close all client connections gracefully
        await asyncio.gather(*(ws.close(code=1001, reason="Server shutting down") for ws in clients))
        server.close()
        await server.wait_closed()
    
    # Clean up resources
    await health_check_site.stop()
    executor.shutdown(wait=True)
    logger.info("Server shut down gracefully.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped manually.")

