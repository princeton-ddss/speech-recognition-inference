from fastapi import FastAPI, HTTPException, status
from fastapi import Response, Request, Header
from fastapi.middleware.cors import CORSMiddleware

import uvicorn
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from api import pipe
from api.models import TranscriptionRequest, TranscriptionResponse, Segment
from api.pipeline import transcribe_audio_file

from api.__init__ import config

app = FastAPI(title="Speech Recognition Inference")

# Set up global rate limit class based on IP Address
limiter = Limiter(
    key_func=get_remote_address, default_limits=["5/10seconds", "10/minute"]
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SlowAPIMiddleware)


@app.get("/", tags=["Speech Recognition Inference"], summary="Welcome message")
def get_root(request: Request):
    return "Welcome to speech-recognition-inference API!"


@app.post(
    "/transcribe",
    response_description="Transcription output",
    summary="Transcribe audio",
    tags=["Speech Recognition Inference"],
)
def run_transcription(
    request: Request, data: TranscriptionRequest, Token: str = Header(None)
) -> TranscriptionResponse:
    """Perform speech-to-text transcription."""
    # Check if authorization is needed
    print("Config Token", config.token)
    print("Input Token", Token)
    if config.token:
        if Token != config.token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
            )
        # else:
        #     #save it to cache so users do not need to pass it again

    audio_path, language, response_format = (
        data.audio_path,
        data.language,
        data.response_format,
    )

    result = transcribe_audio_file(audio_path, pipe, language)
    output = TranscriptionResponse(audio_path=audio_path)
    output.text = result["text"]
    # Whisper determines the language of the whole audio file by looking at
    # first 30 seconds
    output.language = result["chunks"][0]["language"]
    if response_format == "json":
        segments = [None] * len(result["chunks"])
        for idx, seg in enumerate(result["chunks"]):
            segments[idx] = Segment(
                text=seg["text"],
                start=seg["timestamp"][0],
                end=seg["timestamp"][1],
            )
        output.segments = segments

    return output


@app.get(
    "/health",
    response_description="Health check response",
    summary="Health check",
    tags=["Speech Recognition Inference"],
)
@limiter.limit(limit_value="5/10seconds")  # Set up local rate limit on health end
# point
def health_check(request: Request) -> Response:
    return Response()


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8086, reload=True)
