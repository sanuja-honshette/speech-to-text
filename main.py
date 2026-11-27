"""Tiny speech-to-text sidecar. Spring Boot posts raw 16kHz mono int16 PCM here."""
from fastapi import FastAPI, Request

import stt

app = FastAPI(title="Whisper STT sidecar")


@app.on_event("startup")
def warm_up():
    stt.get_model()  # download/load model once at boot


@app.get("/health")
def health():
    return {"status": "ok", "model": stt.MODEL_SIZE}


@app.post("/transcribe")
async def transcribe(request: Request):
    pcm = await request.body()
    text, language = stt.transcribe_pcm(pcm)
    return {"text": text, "language": language}
