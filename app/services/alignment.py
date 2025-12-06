# -*- coding: utf-8 -*-
"""
Optional word-level alignment using wav2vec2 CTC models (WhisperX-style).

References: whisperx/alignment.py
"""
import os
import logging
from typing import List, Optional, Tuple, Dict, Any, Iterable, Union

import numpy as np
import torch
import torchaudio
import librosa
from dataclasses import dataclass
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from app.config.settings import SAMPLE_RATE, ALIGN_MODEL_NAME, USE_ALIGNMENT
from app.utils.logger import logger
from app.services.transcription import load_audio_resampled  # helper we will add

# Default language-specific alignment models (torchaudio bundles first)
DEFAULT_ALIGN_MODELS_TORCH = {
    "en": "WAV2VEC2_ASR_BASE_960H",
    "fr": "VOXPOPULI_ASR_BASE_10K_FR",
    "de": "VOXPOPULI_ASR_BASE_10K_DE",
    "es": "VOXPOPULI_ASR_BASE_10K_ES",
    "it": "VOXPOPULI_ASR_BASE_10K_IT",
}

DEFAULT_ALIGN_MODELS_HF = {
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
    "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
    "ar": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
}


@dataclass
class AlignMetadata:
    language: str
    dictionary: Dict[str, int]
    type: str  # "torchaudio" or "huggingface"
    sample_rate: int


def load_align_model(language_code: str, device: torch.device, model_name: Optional[str] = None, cache_dir: Optional[str] = None):
    """
    Load a wav2vec2 alignment model for the given language.
    """
    if model_name is None:
        if language_code in DEFAULT_ALIGN_MODELS_TORCH:
            model_name = DEFAULT_ALIGN_MODELS_TORCH[language_code]
        elif language_code in DEFAULT_ALIGN_MODELS_HF:
            model_name = DEFAULT_ALIGN_MODELS_HF[language_code]
        else:
            raise ValueError(f"No default alignment model for language: {language_code}")

    if model_name in torchaudio.pipelines.__all__:
        pipeline_type = "torchaudio"
        bundle = torchaudio.pipelines.__dict__[model_name]
        align_model = bundle.get_model(dl_kwargs={"model_dir": cache_dir}).to(device)
        labels = bundle.get_labels()
        align_dictionary = {c.lower(): i for i, c in enumerate(labels)}
        sample_rate = bundle.sample_rate
    else:
        processor = Wav2Vec2Processor.from_pretrained(model_name, cache_dir=cache_dir)
        align_model = Wav2Vec2ForCTC.from_pretrained(model_name, cache_dir=cache_dir)
        pipeline_type = "huggingface"
        align_model = align_model.to(device)
        labels = processor.tokenizer.get_vocab()
        align_dictionary = {char.lower(): code for char, code in processor.tokenizer.get_vocab().items()}
        sample_rate = int(getattr(align_model.config, "sampling_rate", SAMPLE_RATE))

    metadata = AlignMetadata(language=language_code, dictionary=align_dictionary, type=pipeline_type, sample_rate=sample_rate)
    return align_model, processor if pipeline_type == "huggingface" else None, metadata


def _clean_text_for_dict(text: str, dictionary: Dict[str, int], lang_no_spaces: bool = False):
    clean_char: List[str] = []
    clean_idx: List[int] = []
    for idx, ch in enumerate(text):
        ch_ = ch.lower()
        if not lang_no_spaces:
            ch_ = ch_.replace(" ", "|")
        if ch_ in dictionary:
            clean_char.append(ch_)
            clean_idx.append(idx)
        else:
            clean_char.append('*')
            clean_idx.append(idx)
    return clean_char, clean_idx


def _logits_to_timestamps(logits: torch.Tensor, dictionary: Dict[str, int], clean_char: List[str], clean_idx: List[int], frame_shift: float):
    """
    Simplified alignment: greedy CTC decode to align characters back to text positions.
    """
    probs = torch.nn.functional.log_softmax(logits, dim=-1)
    path = torch.argmax(probs, dim=-1).cpu().numpy()

    times: List[Tuple[float, float]] = []
    t_prev = 0.0
    last_char = None
    for t, token in enumerate(path):
        if token == last_char:
            continue
        last_char = token
        if token == dictionary.get("|", -1):
            continue
        if token in dictionary.values():
            times.append((t * frame_shift, (t + 1) * frame_shift))
    # Map times back to text positions (best-effort)
    char_times: Dict[int, Tuple[float, float]] = {}
    for idx, ci in enumerate(clean_idx):
        if idx < len(times):
            char_times[ci] = times[idx]
    return char_times


def align_segments(
    segments: List[Dict[str, Any]],
    audio: Union[str, np.ndarray],
    language: Optional[str],
    device: torch.device,
    model_name: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Align a list of segments (with text/start/end) to word-level timestamps using wav2vec2.
    """
    if not USE_ALIGNMENT:
        return segments

    language = language or "en"

    try:
        align_model, processor, metadata = load_align_model(language, device, model_name=model_name or ALIGN_MODEL_NAME, cache_dir=cache_dir)
    except Exception as e:
        logger.error(f"Failed to load alignment model: {e}", exc_info=True)
        return segments

    lang_no_spaces = language in ["ja", "zh"]

    if isinstance(audio, str):
        waveform = load_audio_resampled(audio, SAMPLE_RATE)
    else:
        waveform = audio
    if waveform.ndim == 1:
        waveform = waveform[None, :]

    frame_shift = 0.02  # approx for wav2vec2 base models

    for seg in segments:
        t1 = seg["start"]
        t2 = seg["end"]
        text = seg.get("text", "")
        if not text or t2 <= t1:
            continue
        f1 = int(t1 * SAMPLE_RATE)
        f2 = int(t2 * SAMPLE_RATE)
        seg_audio = waveform[:, f1:f2]
        with torch.no_grad():
            if metadata.type == "torchaudio":
                emissions, _ = align_model(seg_audio, metadata.sample_rate)
                emissions = emissions[0].cpu()
                logits = emissions
                dictionary = metadata.dictionary
            else:
                inputs = processor(seg_audio.squeeze(0), sampling_rate=metadata.sample_rate, return_tensors="pt")
                logits = align_model(inputs.input_values.to(device)).logits.squeeze(0).cpu()
                dictionary = metadata.dictionary

        clean_char, clean_idx = _clean_text_for_dict(text, dictionary, lang_no_spaces=lang_no_spaces)
        char_times = _logits_to_timestamps(logits, dictionary, clean_char, clean_idx, frame_shift)

        # Assign word-level times best-effort: split on spaces
        words = text.split()
        word_segments = []
        cursor = 0
        for w in words:
            start_idx = text.find(w, cursor)
            end_idx = start_idx + len(w)
            cursor = end_idx
            # find any char time inside this span
            starts = []
            ends = []
            for ci, (cs, ce) in char_times.items():
                if start_idx <= ci < end_idx:
                    starts.append(cs)
                    ends.append(ce)
            if starts and ends:
                w_start = t1 + min(starts)
                w_end = t1 + max(ends)
            else:
                w_start, w_end = t1, t2
            word_segments.append({"word": w + " ", "start": w_start, "end": w_end})
        seg["words"] = word_segments
    return segments

