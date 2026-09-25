"""
=====================================
MODULE 2: Speech to Text Module
=====================================
Converts Urdu speech to text.
Uses: OpenAI Whisper
"""

import os
import whisper
import numpy as np

# Global model (loaded once)
_whisper_model = None
_model_size = "base"


def load_whisper_model(model_size: str = "base"):
    """
    Load Whisper model (once).
    
    Model sizes: tiny, base, small, medium, large
    - tiny   : fastest, less accurate
    - base   : good balance (recommended)
    - small  : slightly better
    - medium : very good for Urdu
    """
    global _whisper_model, _model_size
    if _whisper_model is None or _model_size != model_size:
        print(f"[Speech to Text] Loading Whisper '{model_size}' model...")
        _whisper_model = whisper.load_model(model_size)
        _model_size = model_size
        print(f"[Speech to Text] Model ready!")
    return _whisper_model


def transcribe_audio(file_path: str, language: str = "ur") -> dict:
    """
    Convert audio file to Urdu text.
    
    Args:
        file_path: Path to audio file
        language: Language code ('ur' for Urdu, None for auto-detect)
    
    Returns:
        dict: text, language, segments info
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    model = load_whisper_model()

    print(f"[Speech to Text] Transcribing: {os.path.basename(file_path)}")

    # Transcribe
    options = {
        "fp16": False,  # For running on CPU
        "task": "transcribe"
    }
    if language:
        options["language"] = language

    result = model.transcribe(file_path, **options)

    text = result.get("text", "").strip()
    detected_language = result.get("language", "unknown")
    segments = result.get("segments", [])

    # Confidence calculation (from average no_speech_prob)
    confidence = 1.0
    if segments:
        avg_no_speech = np.mean([s.get("no_speech_prob", 0) for s in segments])
        confidence = round(1.0 - avg_no_speech, 3)

    print(f"[Speech to Text] Result: '{text[:80]}...' " if len(text) > 80 else f"[Speech to Text] Result: '{text}'")
    print(f"[Speech to Text] Language detected: {detected_language}, Confidence: {confidence}")

    return {
        "text": text,
        "language": detected_language,
        "confidence": confidence,
        "word_count": len(text.split()) if text else 0,
        "segments": len(segments)
    }


def speech_to_text(file_path: str) -> str:
    """Simple function - returns text only (backward compatibility)."""
    try:
        result = transcribe_audio(file_path, language="ur")
        return result["text"]
    except Exception as e:
        print(f"[Speech to Text] Error: {e}")
        return ""
