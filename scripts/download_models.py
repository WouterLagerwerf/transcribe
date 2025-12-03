#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-download all required models during Docker build.
This avoids model downloads at container startup.
"""

import os
import sys
import warnings

# Suppress warnings during model loading
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Default model configurations - can be overridden with environment variables
MODEL_NAME = os.getenv("MODEL_NAME", "large-v3")
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
    # Use int8 for CPU (float16 not supported on CPU)
    # Runtime will use float16 if GPU is available
    model = WhisperModel(
        MODEL_NAME,
        device="cpu",
        compute_type="int8",
        download_root=MODEL_PATH
    )
    
    print(f"✅ Whisper model '{MODEL_NAME}' downloaded to {MODEL_PATH}")
    del model


def download_silero_vad():
    """Pre-download Silero VAD model (used by faster-whisper)."""
    print("=" * 80)
    print("📥 Downloading Silero VAD model")
    print("=" * 80)
    
    import torch
    
    # Download Silero VAD model - this is what faster-whisper uses internally
    try:
        torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        print("✅ Silero VAD model downloaded")
    except Exception as e:
        print(f"⚠️  Failed to download Silero VAD: {e}")
        print("   faster-whisper will download it on first use")


def download_speaker_embedding_model():
    """Download and cache the pyannote speaker embedding model."""
    print("=" * 80)
    print("📥 Downloading speaker embedding model")
    print("=" * 80)
    
    if not HF_TOKEN:
        print("   ⚠️  No HF_TOKEN provided - speaker embedding model requires authentication")
        print("   ⚠️  Set HF_TOKEN build arg to download speaker embedding model")
        print("   ⚠️  Speaker identification will be disabled at runtime")
        return False
    
    os.environ['HF_TOKEN'] = HF_TOKEN
    os.environ['HUGGING_FACE_HUB_TOKEN'] = HF_TOKEN
    print(f"   Using HF_TOKEN (length: {len(HF_TOKEN)})")
    
    try:
        # Login to HuggingFace
        from huggingface_hub import login
        login(token=HF_TOKEN, add_to_git_credential=False)
        
        # Download embedding model
        print("\n📥 Downloading pyannote/embedding...")
        
        # Suppress PyTorch Lightning warnings
        import logging
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
        logging.getLogger("lightning_fabric").setLevel(logging.ERROR)
        
        from pyannote.audio import Model
        
        # Load model to cache it
        embedding_model = Model.from_pretrained(
            "pyannote/embedding",
            token=HF_TOKEN
        )
        
        # Verify it loaded correctly
        if embedding_model is not None:
            # Run a quick test to make sure it's cached properly
            embedding_model.eval()
            print("✅ Speaker embedding model downloaded and verified")
            del embedding_model
            return True
        else:
            print("⚠️  Model loaded as None - may need to re-download at runtime")
            return False
        
    except Exception as e:
        print(f"⚠️  Failed to download speaker embedding model: {e}")
        print("   Make sure you have accepted the model terms at:")
        print("   - https://huggingface.co/pyannote/embedding")
        return False


def main():
    print("\n" + "=" * 80)
    print("🚀 PRE-DOWNLOADING ALL MODELS")
    print("=" * 80 + "\n")
    
    success = True
    
    # Download Whisper model (required)
    try:
        download_whisper_model()
    except Exception as e:
        print(f"❌ Failed to download Whisper model: {e}")
        sys.exit(1)
    
    # Download Silero VAD (used by faster-whisper)
    try:
        download_silero_vad()
    except Exception as e:
        print(f"⚠️  Failed to download Silero VAD: {e}")
        # Not fatal - faster-whisper will download it
    
    # Download speaker embedding model (optional - requires HF token)
    try:
        speaker_success = download_speaker_embedding_model()
        if not speaker_success:
            print("\n   ℹ️  Speaker identification will be disabled without the embedding model")
    except Exception as e:
        print(f"⚠️  Failed to download speaker embedding model: {e}")
        print("   Speaker identification will be disabled at runtime")
    
    print("\n" + "=" * 80)
    print("✅ MODEL PRE-DOWNLOAD COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
