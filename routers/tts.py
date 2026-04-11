import os
import httpx
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

router = APIRouter(tags=["TTS"])

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
DEEPGRAM_TTS_URL = "https://api.deepgram.com/v1/speak"

@router.get("/api/tts")
async def text_to_speech(text: str = Query(...)):
    async def stream_audio():
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{DEEPGRAM_TTS_URL}?model=aura-2-phoebe-en",
                headers={
                    "Authorization": f"Token {DEEPGRAM_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={"text": text},
                timeout=30,
            ) as response:
                async for chunk in response.aiter_bytes():
                    yield chunk

    return StreamingResponse(stream_audio(), media_type="audio/mpeg")