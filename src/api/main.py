from fastapi import FastAPI
from fastapi import Response, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from api import pipe
from api.models import TranscriptionRequest, TranscriptionResponse, Segment
from api.pipeline import transcribe_audio_file

app = FastAPI(title="Speech Recognition Inference")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Set up limiter class based on IP Address
limiter = Limiter(key_func=get_remote_address,
    default_limits=["2/10seconds", "10/minute"]) #Could customize based on
# users in the request
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.get("/", tags=["Speech Recognition Inference"], summary="Welcome message")
@limiter.limit(limit_value="2/5seconds; 10/minute")
def get_root(request:Request):
    return "Welcome to speech-recognition-inference API!"


@app.post(
    "/transcribe",
    response_description="Transcription output",
    summary="Transcribe audio",
    tags=["Speech Recognition Inference"],
)
@limiter.limit(limit_value="2/5seconds; 10/minute")
def run_transcription(request:Request, data: TranscriptionRequest) -> \
        TranscriptionResponse:
    """Perform speech-to-text transcription."""

    audio_path, language, response_format = (
        data.audio_path,
        data.language,
        data.response_format,
    )

    result = transcribe_audio_file(audio_path, pipe, language)
    output = TranscriptionResponse(audio_path=audio_path)
    output.text = result["text"]
    if response_format == "json":
        segments = [None] * len(result["chunks"])
        for idx, seg in enumerate(result["chunks"]):
            segments[idx] = Segment(
                language=seg["language"],
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
@limiter.limit(limit_value="5/10seconds") #Set higher limit on health end
# point
def health_check(request:Request) -> Response:
    return Response()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8085, reload = True)