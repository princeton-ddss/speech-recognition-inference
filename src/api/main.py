from fastapi import FastAPI
from fastapi import Response
from fastapi.middleware.cors import CORSMiddleware

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


@app.get("/", tags=["Speech Recognition Inference"], summary="Welcome message")
def get_root():
    return "Welcome to speech-recognition-inference API!"


@app.post(
    "/transcribe",
    response_description="Transcription output",
    summary="Transcribe audio",
    tags=["Speech Recognition Inference"],
)
def run_transcription(data: TranscriptionRequest) -> TranscriptionResponse:
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
def health_check() -> Response:
    return Response()
