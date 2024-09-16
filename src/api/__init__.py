import os
import argparse
from fastapi import FastAPI

from api.config import SpeechRecognitionInferenceConfig
from api.models import TranscriptionRequest, TranscriptionResponse, Segment
from api.pipeline import load_pipeline, transcribe_audio_file
from api.logger import logger


parser = argparse.ArgumentParser()
parser.add_argument("--model_id", help="The model to use, e.g., openai/whisper-tiny")
parser.add_argument("--revision_id", help="The model revision to use, e.g., b5d...")
parser.add_argument(
    "--model_dir",
    help="The location to look for model files, e.g., ~/.cache/huggingface/hub",
)
parser.add_argument(
    "--audio_dir", help="The location to look for audio files, e.g., /tmp"
)
parser.add_argument(
    "--port", type=int, help="The port to serve the API on (default: 8000)."
)
parser.add_argument("--host", help="The host to serve the API on (default: 127.0.0.1).")
parser.add_argument(
    "--reload",
    type=bool,
    help="Automatically reload after changes to the source code (for development).",
    default=False,
)
args = parser.parse_args()

config = SpeechRecognitionInferenceConfig()

if os.getenv("SRI_MODEL_DIR") is not None:
    config.model_dir = os.getenv("SRI_MODEL_DIR")
elif args.model_dir is not None:
    config.model_dir = args.model_dir

if config.model_dir is None:
    raise Exception(
        "A model directory is required. Either set the SRI_AUDIO_DIR environment"
        " variables or pass in a value for --model_dir"
    )
if not os.path.isdir(config.model_dir):
    raise Exception(f"model_dir {config.model_dir} does not exist. ")

if os.getenv("SRI_AUDIO_DIR") is not None:
    config.audio_dir = os.getenv("SRI_AUDIO_DIR")
elif args.audio_dir is not None:
    config.audio_dir = args.audio_dir

if config.audio_dir is None:
    raise Exception(
        "An audio directory is required. Either set the SRI_AUDIO_DIR environment"
        " variables or pass in a value for --audio_dir"
    )
if not os.path.isdir(config.audio_dir):
    raise Exception(f"audio_dir {config.audio_dir} does not exist. ")

if os.getenv("SRI_PORT") is not None:
    config.port = int(os.getenv("SRI_PORT"))
elif args.port is not None:
    config.port = args.port

if os.getenv("SRI_HOST") is not None:
    config.host = os.getenv("SRI_HOST")
elif args.host is not None:
    config.host = args.host

if args.model_id is not None:
    config.model_id = args.model_id

if args.revision_id is not None:
    config.revision_id = args.revision_id

config.auto_reload = args.reload

hf_access_token = os.getenv("HF_ACCESS_TOKEN", None)

logger.info(
    f"Initializing speech_recognition_inference service (model_dir={config.model_dir},"
    f" revision_id={config.revision_id}, model_id={config.model_id},"
    f" audio_dir={config.audio_dir}, hf_access_token={hf_access_token})"
)

pipe = load_pipeline(config.model_dir, config.model_id, config.revision_id)

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

    audio_file, language, response_format = (
        data.audio_file,
        data.language,
        data.response_format,
    )

    audio_path = os.path.join(config.audio_dir, audio_file)
    result = transcribe_audio_file(audio_path, pipe, language)
    output = TranscriptionResponse(audio_file=audio_file)
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
