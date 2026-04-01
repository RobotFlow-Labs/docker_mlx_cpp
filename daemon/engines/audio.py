"""
Audio Engine — MLX-powered speech-to-text and text-to-speech.

Supports:
    - Whisper STT (mlx-audio / mlx-community whisper models)
    - TTS with voice selection (mlx-audio)

OpenAI-compatible /v1/audio/transcriptions and /v1/audio/speech endpoints.
"""

import io
import logging
import tempfile
import time
from pathlib import Path

logger = logging.getLogger("mlx-daemon.audio")


def transcribe(audio_bytes: bytes, model_id: str = "mlx-community/whisper-large-v3-turbo") -> dict:
    """Transcribe audio to text using Whisper via MLX."""
    start = time.monotonic()

    try:
        from mlx_audio.stt import transcribe as mlx_transcribe

        # Write audio to temp file (mlx-audio expects file path)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name

        result = mlx_transcribe(temp_path, model=model_id)

        # Clean up
        Path(temp_path).unlink(missing_ok=True)

        elapsed = time.monotonic() - start
        logger.info("Transcription: %s, %.1fs", model_id, elapsed)

        return {
            "text": result.get("text", ""),
            "model": model_id,
            "duration_ms": int(elapsed * 1000),
        }

    except ImportError:
        logger.error("mlx-audio not installed. Install with: pip install 'docker-mlx-cpp[audio]'")
        return {"error": "mlx-audio not installed. Install with: pip install 'docker-mlx-cpp[audio]'"}
    except Exception as e:
        logger.error("Transcription failed: %s", e)
        return {"error": str(e)}


def text_to_speech(text: str, voice: str = "alloy") -> bytes:
    """Convert text to speech using MLX TTS."""
    try:
        from mlx_audio.tts import generate as mlx_tts_generate

        start = time.monotonic()

        # Generate audio
        audio_data = mlx_tts_generate(text, voice=voice)

        elapsed = time.monotonic() - start
        logger.info("TTS: %d chars, voice=%s, %.1fs", len(text), voice, elapsed)

        # Convert to bytes (wav/mp3)
        if isinstance(audio_data, bytes):
            return audio_data

        # If numpy array, convert to wav
        import numpy as np
        import wave

        buf = io.BytesIO()
        if isinstance(audio_data, np.ndarray):
            audio_int16 = (audio_data * 32767).astype(np.int16)
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(22050)
                wf.writeframes(audio_int16.tobytes())
            return buf.getvalue()

        return b""

    except ImportError:
        logger.error("mlx-audio not installed for TTS")
        return b""
    except Exception as e:
        logger.error("TTS failed: %s", e)
        return b""
