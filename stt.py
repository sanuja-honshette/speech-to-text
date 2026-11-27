"""Streaming speech-to-text using faster-whisper (open source, self-hosted).

Whisper's multilingual model handles Hindi + English code-switching. We leave
`language=None` so it auto-detects per chunk, and use an initial prompt written
in Hinglish to bias it toward Devanagari/Latin mixed output.
"""
import os
import threading

import numpy as np

SAMPLE_RATE = 16000
MODEL_SIZE = os.getenv("WHISPER_MODEL", "small")  # tiny/base/small/medium
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE", "int8")
LANGUAGE = os.getenv("WHISPER_LANGUAGE") or None  # None = auto-detect
INITIAL_PROMPT = "Namaste, मैं आज meeting में project के बारे में बात करूँगा. Hello, कैसे हैं आप?"

_model = None
_lock = threading.Lock()


def get_model():
    global _model
    with _lock:
        if _model is None:
            from faster_whisper import WhisperModel

            _model = WhisperModel(MODEL_SIZE, device="cpu", compute_type=COMPUTE_TYPE)
    return _model


def transcribe_pcm(pcm16: bytes):
    """Transcribe raw 16kHz mono int16 PCM. Returns (text, language)."""
    if len(pcm16) < SAMPLE_RATE * 2 * 0.3:  # < 0.3s: nothing to do
        return "", None
    audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
    segments, info = get_model().transcribe(
        audio,
        language=LANGUAGE,
        task="transcribe",
        beam_size=1,
        vad_filter=True,
        initial_prompt=INITIAL_PROMPT,
        condition_on_previous_text=False,
    )
    text = " ".join(s.text.strip() for s in segments).strip()
    return text, info.language
