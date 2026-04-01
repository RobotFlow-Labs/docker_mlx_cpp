"""
Audio Engine — MLX-powered speech-to-text and text-to-speech.

STT: Whisper via mlx-audio (generate_transcription API)
TTS: Kokoro via mlx-audio (load_model + stream generate API)

OpenAI-compatible /v1/audio/transcriptions and /v1/audio/speech endpoints.
"""

import io
import logging
import tempfile
import time
from pathlib import Path

logger = logging.getLogger("mlx-daemon.audio")

_tts_models: dict[str, object] = {}


def transcribe(audio_bytes: bytes, model_id: str = "mlx-community/whisper-large-v3-turbo-asr-fp16") -> dict:
    """Transcribe audio to text using Whisper via MLX."""
    start = time.monotonic()

    try:
        from mlx_audio.stt.generate import generate_transcription

        # Write audio to temp file (mlx-audio expects file path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name

        result = generate_transcription(model=model_id, audio=temp_path)

        # Clean up temp file
        Path(temp_path).unlink(missing_ok=True)

        elapsed = time.monotonic() - start
        logger.info("Transcription: %s, %.1fs", model_id, elapsed)

        return {
            "text": result.text,
            "model": model_id,
            "duration_ms": int(elapsed * 1000),
        }

    except ImportError:
        logger.error("mlx-audio not installed. pip install 'docker-mlx-cpp[audio]'")
        return {"error": "mlx-audio not installed. Install with: pip install 'docker-mlx-cpp[audio]'"}
    except Exception as e:
        logger.error("Transcription failed: %s", e)
        return {"error": str(e)}


def text_to_speech(
    text: str,
    voice: str = "af_heart",
    model_id: str = "mlx-community/Kokoro-82M-bf16",
) -> bytes:
    """Convert text to speech using MLX TTS (Kokoro)."""
    try:
        from mlx_audio.tts.utils import load_model
        import numpy as np
        import wave

        start = time.monotonic()

        if model_id not in _tts_models:
            logger.info("Loading TTS model %s...", model_id)
            _tts_models[model_id] = load_model(model_id)

        tts_model = _tts_models[model_id]

        # Collect audio segments from streaming generator
        audio_segments = []
        sample_rate = 24000  # Kokoro default

        for result in tts_model.generate(text=text, voice=voice, speed=1.0):
            audio_segments.append(np.array(result.audio))
            if hasattr(result, "sample_rate"):
                sample_rate = result.sample_rate

        if not audio_segments:
            logger.warning("TTS produced no audio for: %s", text[:50])
            return b""

        # Concatenate all segments
        audio_np = np.concatenate(audio_segments)

        # Convert to WAV bytes
        buf = io.BytesIO()
        audio_int16 = (audio_np * 32767).astype(np.int16)
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())

        elapsed = time.monotonic() - start
        logger.info("TTS: %d chars, voice=%s, %.1fs", len(text), voice, elapsed)
        return buf.getvalue()

    except ImportError:
        logger.error("mlx-audio not installed. pip install 'docker-mlx-cpp[audio]'")
        return b""
    except Exception as e:
        logger.error("TTS failed: %s", e)
        return b""
