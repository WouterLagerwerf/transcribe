"""HTTP API handler for transcription."""

import asyncio
import numpy as np
from aiohttp import web
from aiohttp.web_request import Request

from app.config.settings import SAMPLE_RATE, USE_DIARIZATION
from app.utils.logger import logger
from app.services.transcription import transcribe_synchronous, get_executor, is_server_ready
from app.services.speaker_identification import (
    SpeakerIdentifier, is_model_loaded as is_speaker_model_loaded
)
from app.services.diarization import (
    diarize_audio,
    is_diarization_model_loaded
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
        
        # Perform speaker identification + overlap diarization if enabled
        has_speakers = False
        if USE_DIARIZATION and is_speaker_model_loaded():
            logger.info("Performing speaker identification (overlap-aware if diarization is loaded)...")

            identifier = SpeakerIdentifier()

            diar_regions = []
            if is_diarization_model_loaded():
                diar_regions = await loop.run_in_executor(executor, diarize_audio, audio_float32)

            mapped_regions = []
            if diar_regions and is_diarization_model_loaded():
                for region in diar_regions:
                    seg_start = max(0, int(region["start"] * SAMPLE_RATE))
                    seg_end = min(len(audio_float32), int(region["end"] * SAMPLE_RATE))
                    if seg_end <= seg_start:
                        continue
                    segment_audio = audio_float32[seg_start:seg_end]
                    speaker_label, confidence = await loop.run_in_executor(
                        executor, identifier.identify_speaker, segment_audio
                    )
                    if speaker_label:
                        mapped_regions.append({
                            "start": region["start"],
                            "end": region["end"],
                            "speaker": speaker_label,
                            "score": confidence
                        })

            def pick_best_speaker(span_start: float, span_end: float) -> str:
                best_label = None
                best_overlap = 0.0
                for reg in mapped_regions:
                    overlap = min(span_end, reg["end"]) - max(span_start, reg["start"])
                    if overlap > best_overlap and overlap > 0:
                        best_overlap = overlap
                        best_label = reg["speaker"]
                return best_label

            for segment in segments:
                seg_start = int(segment["start"] * SAMPLE_RATE)
                seg_end = int(segment["end"] * SAMPLE_RATE)
                seg_start = max(0, seg_start)
                seg_end = min(len(audio_float32), seg_end)

                if seg_end <= seg_start:
                    continue

                # If we have diarization, use word-level assignment; otherwise embedding per segment
                if mapped_regions:
                    words = segment.get("words") or []
                    if words:
                        # Assign speaker per word and group
                        grouped = []
                        current_words = []
                        current_speaker = None
                        for w in words:
                            w_speaker = pick_best_speaker(w.get("start", segment["start"]), w.get("end", segment["end"]))
                            if current_speaker is None:
                                current_speaker = w_speaker
                            if w_speaker != current_speaker and current_words:
                                text = "".join([cw["word"] for cw in current_words]).strip()
                                grouped.append({
                                    "text": text,
                                    "start": current_words[0]["start"],
                                    "end": current_words[-1]["end"],
                                    "speaker": current_speaker
                                })
                                current_words = []
                            current_speaker = w_speaker
                            current_words.append(w)
                        if current_words:
                            text = "".join([cw["word"] for cw in current_words]).strip()
                            grouped.append({
                                "text": text,
                                "start": current_words[0]["start"],
                                "end": current_words[-1]["end"],
                                "speaker": current_speaker
                            })

                        # Replace original segment with grouped sub-segments
                        segment["sub_segments"] = grouped
                        if any(g.get("speaker") for g in grouped):
                            has_speakers = True
                    else:
                        speaker_label = pick_best_speaker(segment["start"], segment["end"])
                        if speaker_label:
                            segment["speaker"] = speaker_label
                            has_speakers = True
                else:
                    # No diarization pipeline; fall back to embedding on the whole segment
                    segment_audio = audio_float32[seg_start:seg_end]
                    speaker_label, confidence = await loop.run_in_executor(
                        executor, identifier.identify_speaker, segment_audio
                    )
                    if speaker_label:
                        segment["speaker"] = speaker_label
                        has_speakers = True

            speaker_count = identifier.get_speaker_count()
            logger.info(f"Speaker identification completed: {speaker_count} speakers identified")
        
        # Flatten sub-segments (overlap-aware) if present
        output_segments = []
        for seg in segments:
            if seg.get("sub_segments"):
                for sub in seg["sub_segments"]:
                    output_segments.append({
                        "text": sub["text"],
                        "start": float(sub["start"]),
                        "end": float(sub["end"]),
                        **({"speaker": sub["speaker"]} if sub.get("speaker") else {})
                    })
            else:
                output_segments.append({
                    "text": seg["text"],
                    "start": float(seg["start"]),
                    "end": float(seg["end"]),
                    **({"speaker": seg["speaker"]} if seg.get("speaker") else {})
                })
        
        # Return segments with timestamps (and speaker info if identification was performed)
        return web.json_response({
            "segments": output_segments,
            "total_segments": len(output_segments),
            "full_text": " ".join([s["text"] for s in output_segments]).strip(),
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
