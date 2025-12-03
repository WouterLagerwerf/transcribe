"""HTTP API server for health checks and transcription."""

from aiohttp import web

from app.config.settings import SERVER_HOST, HEALTH_CHECK_PORT, USE_DIARIZATION
from app.utils.logger import logger
from app.services.transcription import is_server_ready, get_model_name
from app.services.speaker_identification import is_model_loaded as is_speaker_model_loaded
from app.handlers.transcription_api import transcribe_handler
from app.handlers.websocket_handler import get_session_count


async def health_check_handler(request):
    """HTTP handler for health checks."""
    if is_server_ready():
        response_data = {
            "status": "ok",
            "model": get_model_name(),
            "speaker_identification_enabled": USE_DIARIZATION,
            "speaker_identification_loaded": is_speaker_model_loaded() if USE_DIARIZATION else False,
            "active_sessions": get_session_count()
        }
        return web.json_response(response_data)
    else:
        return web.json_response(
            {"status": "unavailable", "reason": "Model not loaded"},
            status=503
        )


async def start_http_server():
    """Starts the aiohttp server for health checks and transcription API."""
    app = web.Application()
    app.router.add_get("/health", health_check_handler)
    app.router.add_post("/transcribe", transcribe_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, SERVER_HOST, HEALTH_CHECK_PORT)
    await site.start()
    logger.info(f"HTTP API server running on http://{SERVER_HOST}:{HEALTH_CHECK_PORT}")
    logger.info(f"  POST /transcribe - Upload audio file for transcription")
    logger.info(f"  GET  /health - Health check endpoint")
    return site
