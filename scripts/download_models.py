#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-download all required models during Docker build.
This avoids model downloads at container startup.
"""

import os
import sys

# Default model configurations - can be overridden with environment variables
MODEL_NAME = os.getenv("MODEL_NAME", "large-v3")  # Default to large-v3 for best quality
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models")
HF_TOKEN = os.getenv("HF_TOKEN", None)


def download_whisper_model():
    """Download and cache the faster-whisper model."""
    print("=" * 80)
    print(f"📥 Downloading Whisper model: {MODEL_NAME}")
    print("=" * 80)
    
    from faster_whisper import WhisperModel
    
    # Create cache directory
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    # Download model - use CPU for download phase (no GPU during build)
    # The model files are the same regardless of device
    model = WhisperModel(
        MODEL_NAME,
        device="cpu",
        compute_type="int8",
        download_root=MODEL_PATH
    )
    
    print(f"✅ Whisper model '{MODEL_NAME}' downloaded to {MODEL_PATH}")
    del model


def download_speaker_embedding_model():
    """Download and cache the pyannote speaker embedding model."""
    print("=" * 80)
    print("📥 Downloading speaker embedding model")
    print("=" * 80)
    
    if HF_TOKEN:
        os.environ['HF_TOKEN'] = HF_TOKEN
        os.environ['HUGGING_FACE_HUB_TOKEN'] = HF_TOKEN
        print(f"   Using HF_TOKEN (length: {len(HF_TOKEN)})")
    else:
        print("   ⚠️  No HF_TOKEN provided - speaker embedding model requires authentication")
        print("   ⚠️  Set HF_TOKEN build arg to download speaker embedding model")
        return
    
    try:
        from huggingface_hub import login
        login(token=HF_TOKEN, add_to_git_credential=False)
        
        # Download embedding model (used for speaker identification)
        print("\n📥 Downloading pyannote/embedding...")
        from pyannote.audio import Model
        embedding_model = Model.from_pretrained(
            "pyannote/embedding",
            token=HF_TOKEN
        )
        print("✅ Speaker embedding model downloaded")
        del embedding_model
        
    except Exception as e:
        print(f"⚠️  Failed to download speaker embedding model: {e}")
        print("   Make sure you have accepted the model terms at:")
        print("   - https://huggingface.co/pyannote/embedding")


def main():
    print("\n" + "=" * 80)
    print("🚀 PRE-DOWNLOADING ALL MODELS")
    print("=" * 80 + "\n")
    
    # Download Whisper model (required)
    # Note: faster-whisper downloads Silero VAD automatically when vad_filter=True
    try:
        download_whisper_model()
    except Exception as e:
        print(f"❌ Failed to download Whisper model: {e}")
        sys.exit(1)
    
    # Download speaker embedding model (optional - requires HF token)
    try:
        download_speaker_embedding_model()
    except Exception as e:
        print(f"⚠️  Failed to download speaker embedding model: {e}")
        print("   Speaker identification will be disabled if model not available at runtime")
    
    print("\n" + "=" * 80)
    print("✅ MODEL PRE-DOWNLOAD COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
