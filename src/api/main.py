from fastapi import FastAPI

from api import pipe
from api.models import TranscriptionRequest, TranscriptionResponse, Segment
from api.pipeline import transcribe_audio_file


app = FastAPI()


@app.get("/")
def get_root():
    return "Welcome to speech-recognition-inference API!"


@app.post(
    "/transcribe",
    tags=["transcription"],
    response_description="Transcription Outputs",
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
