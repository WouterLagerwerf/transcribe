"""Text correction service using a small LLM for context-aware grammar and logic fixes."""

import os
from typing import List, Optional
from app.utils.logger import logger

# Try to import LLM libraries
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama not available. Text correction will use fallback methods.")

# Ollama host configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Global state
correction_enabled = False
correction_model = None
recent_sentences = []  # Sliding window of recent sentences for context
MAX_CONTEXT_SENTENCES = 3


def load_correction_model():
    """Load the text correction model."""
    global correction_enabled, correction_model
    
    use_correction = os.getenv("USE_TEXT_CORRECTION", "false").lower() == "true"
    if not use_correction:
        correction_enabled = False
        logger.info("Text correction is disabled.")
        return
    
    correction_model_name = os.getenv("CORRECTION_MODEL", "llama3.2:1b")  # Small, fast model
    
    if OLLAMA_AVAILABLE:
        try:
            # Configure Ollama client to use the correct host
            import ollama
            client = ollama.Client(host=OLLAMA_HOST)
            
            # Check if Ollama is running and model is available
            try:
                logger.info(f"Connecting to Ollama at {OLLAMA_HOST}...")
                models_response = client.list()  # Test connection and get available models
                logger.debug(f"Ollama connection successful. Response type: {type(models_response)}, Response: {models_response}")
                
                # Extract model names from response
                # Ollama client.list() returns a ListResponse object with a 'models' attribute
                # Each model in the list has a 'name' attribute
                models_list = models_response.models if hasattr(models_response, 'models') else []
                
                # Extract model names
                model_names = []
                for m in models_list:
                    if hasattr(m, 'name'):
                        model_names.append(m.name)
                    elif isinstance(m, dict):
                        name = m.get('name', '')
                        if name:
                            model_names.append(name)
                    else:
                        logger.warning(f"Unexpected model format: {m} (type: {type(m)})")
                
                logger.info(f"Available Ollama models: {model_names}")
                
                # Check if the model is available, if not try to pull it
                if correction_model_name not in model_names:
                    logger.info(f"Model {correction_model_name} not found, pulling...")
                    try:
                        for chunk in client.pull(correction_model_name, stream=True):
                            if chunk.get('status'):
                                logger.info(f"Pulling model: {chunk.get('status')}")
                        logger.info(f"Successfully pulled model {correction_model_name}")
                    except Exception as pull_error:
                        logger.warning(f"Failed to pull model {correction_model_name}: {pull_error}")
                        logger.warning("Text correction will be disabled until model is available.")
                        correction_enabled = False
                        return
                
                correction_enabled = True
                correction_model = correction_model_name
                logger.info(f"✅ Text correction enabled using Ollama model: {correction_model_name} at {OLLAMA_HOST}")
            except Exception as e:
                logger.error(f"Ollama not available at {OLLAMA_HOST}: {e}", exc_info=True)
                logger.warning("Text correction disabled.")
                correction_enabled = False
        except Exception as e:
            logger.warning(f"Failed to initialize Ollama: {e}. Text correction disabled.")
            correction_enabled = False
    else:
        logger.warning("Ollama not installed. Install with: pip install ollama")
        correction_enabled = False


def correct_text_with_llm(text: str, context_sentences: List[str]) -> str:
    """
    Use LLM to correct text based on context.
    
    Args:
        text: The text to correct
        context_sentences: List of previous sentences for context
    
    Returns:
        Corrected text
    """
    if not correction_enabled or not OLLAMA_AVAILABLE:
        logger.debug(f"Text correction disabled, returning original text: '{text}'")
        return text
    
    try:
        # Build context prompt
        context = "\n".join(context_sentences[-MAX_CONTEXT_SENTENCES:]) if context_sentences else ""
        
        prompt = f"""You are a text correction assistant. Your job is to fix grammatical errors, typos, and make sentences more logical and natural, especially for Dutch and English.

Previous context:
{context}

Text to correct: "{text}"

Return ONLY the corrected text, nothing else. Keep the meaning the same but fix grammar, spelling, and make it sound natural."""

        # Use configured Ollama host
        logger.info(f"📞 Calling Ollama at {OLLAMA_HOST} with model {correction_model} for text: '{text[:50]}...'")
        client = ollama.Client(host=OLLAMA_HOST)
        response = client.generate(
            model=correction_model,
            prompt=prompt,
            options={
                "temperature": 0.3,  # Low temperature for consistent corrections
                "num_predict": 100,  # Limit response length
            }
        )
        
        corrected = response.get("response", text).strip()
        
        # Remove quotes if the model wrapped the response
        if corrected.startswith('"') and corrected.endswith('"'):
            corrected = corrected[1:-1]
        
        if corrected != text:
            logger.info(f"Text correction: '{text}' -> '{corrected}'")
        else:
            logger.debug(f"No correction needed: '{text}'")
        
        return corrected
    except Exception as e:
        logger.error(f"Text correction failed: {e}", exc_info=True)
        logger.warning(f"Returning original text: '{text}'")
        return text


def correct_text_simple(text: str, context_sentences: List[str]) -> str:
    """
    Simple rule-based correction as fallback.
    This can be expanded with common patterns.
    """
    # Basic fixes
    corrected = text
    
    # Fix common Dutch patterns
    corrections = {
        "net wat": "nou daadwerkelijk",
        "wat werkelijk": "daadwerkelijk",
        "het net": "het nou",
        # Add more patterns as needed
    }
    
    for wrong, correct in corrections.items():
        if wrong in corrected.lower():
            corrected = corrected.replace(wrong, correct)
            logger.debug(f"Applied correction: '{wrong}' -> '{correct}'")
    
    return corrected


def correct_transcription_segment(text: str, update_context: bool = True) -> str:
    """
    Correct a transcription segment using context from previous sentences.
    
    Args:
        text: The text segment to correct
        update_context: Whether to add this sentence to the context window
    
    Returns:
        Corrected text
    """
    global recent_sentences
    
    if not text or not text.strip():
        return text
    
    logger.info(f"🔧 correct_transcription_segment called: '{text}' (correction_enabled: {correction_enabled}, OLLAMA_AVAILABLE: {OLLAMA_AVAILABLE})")
    
    # Get context from recent sentences
    context = recent_sentences[-MAX_CONTEXT_SENTENCES:] if recent_sentences else []
    logger.debug(f"Context sentences: {context}")
    
    # Correct the text
    if correction_enabled:
        logger.info(f"🔧 Calling LLM correction for: '{text}'")
        corrected = correct_text_with_llm(text, context)
    else:
        logger.info(f"🔧 Using simple correction (LLM disabled) for: '{text}'")
        corrected = correct_text_simple(text, context)
    
    # Update context window
    if update_context and corrected:
        recent_sentences.append(corrected)
        # Keep only the last N sentences
        if len(recent_sentences) > MAX_CONTEXT_SENTENCES * 2:
            recent_sentences = recent_sentences[-MAX_CONTEXT_SENTENCES * 2:]
    
    return corrected


def reset_context():
    """Reset the context window (useful when starting a new conversation)."""
    global recent_sentences
    recent_sentences = []
    logger.debug("Text correction context reset")


def is_correction_enabled() -> bool:
    """Returns whether text correction is enabled."""
    return correction_enabled

