"""
Vosk transcription service.
- Model is loaded once at startup (heavy — ~200MB for small, ~1.8GB for large).
- Each WebSocket session gets its own KaldiRecognizer (stateful, not thread-safe).
- VOSK_MODEL_PATH env var points to the unzipped model directory.
"""
import json
import logging
import os
from typing import Optional, Tuple

from vosk import Model, KaldiRecognizer

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
_model: Optional[Model] = None


def load_model() -> None:
    """Called once at FastAPI startup. Crashes early if model path is wrong."""
    global _model
    model_path = os.getenv("VOSK_MODEL_PATH", "vosk-model-small-en-us-0.15")
    if not os.path.exists(model_path):
        raise RuntimeError(
            f"Vosk model not found at '{model_path}'. "
            f"Download from https://alphacephei.com/vosk/models and set VOSK_MODEL_PATH."
        )
    logger.info(f"Loading Vosk model from {model_path}...")
    _model = Model(model_path)
    logger.info("Vosk model loaded successfully.")


def get_model() -> Model:
    if _model is None:
        raise RuntimeError("Vosk model not loaded. Call load_model() at startup.")
    return _model


def create_recognizer() -> KaldiRecognizer:
    """Create a fresh recognizer for a new session."""
    rec = KaldiRecognizer(get_model(), SAMPLE_RATE)
    rec.SetWords(True)
    return rec


def process_chunk(recognizer: KaldiRecognizer, pcm_bytes: bytes) -> Tuple[Optional[str], Optional[str]]:
    """
    Feed a raw PCM chunk to the recognizer.

    Returns:
        (partial_text, final_text) — at most one will be non-None per call.
        final_text is set when Vosk detects end-of-utterance (silence).
    """
    if recognizer.AcceptWaveform(pcm_bytes):
        result = json.loads(recognizer.Result())
        text = result.get("text", "").strip()
        print("final", text)
        return None, text if text else None
    else:
        partial = json.loads(recognizer.PartialResult())
        text = partial.get("partial", "").strip()
        print("partial", text)
        return text if text else None, None


def flush_recognizer(recognizer: KaldiRecognizer) -> Optional[str]:
    """Flush any remaining audio when the session ends."""
    result = json.loads(recognizer.FinalResult())
    text = result.get("text", "").strip()
    return text if text else None