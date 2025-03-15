import os
from typing import Optional, Union, Literal
from dataclasses import dataclass

import uvicorn
import typer
from pydantic import BaseModel, Field, ConfigDict
from fastapi import FastAPI, HTTPException, status
from fastapi import Response, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, Pipeline, pipeline
from utils import download_hf_models, get_latest_commit

from logger import logger


class TranscriptionRequest(BaseModel):
    audio_path: str = Field(examples=["/data/audio.wav"])
    language: Literal[
        "english",
        "chinese",
        "german",
        "spanish",
        "russian",
        "korean",
        "french",
        "japanese",
        "portuguese",
        "turkish",
        "polish",
        "catalan",
        "dutch",
        "arabic",
        "swedish",
        "italian",
        "indonesian",
        "hindi",
        "finnish",
        "vietnamese",
        "hebrew",
        "ukrainian",
        "greek",
        "malay",
        "czech",
        "romanian",
        "danish",
        "hungarian",
        "tamil",
        "norwegian",
        "thai",
        "urdu",
        "croatian",
        "bulgarian",
        "lithuanian",
        "latin",
        "maori",
        "malayalam",
        "welsh",
        "slovak",
        "telugu",
        "persian",
        "latvian",
        "bengali",
        "serbian",
        "azerbaijani",
        "slovenian",
        "kannada",
        "estonian",
        "macedonian",
        "breton",
        "basque",
        "icelandic",
        "armenian",
        "nepali",
        "mongolian",
        "bosnian",
        "kazakh",
        "albanian",
        "swahili",
        "galician",
        "marathi",
        "punjabi",
        "sinhala",
        "khmer",
        "shona",
        "yoruba",
        "somali",
        "afrikaans",
        "occitan",
        "georgian",
        "belarusian",
        "tajik",
        "sindhi",
        "gujarati",
        "amharic",
        "yiddish",
        "lao",
        "uzbek",
        "faroese",
        "haitian creole",
        "pashto",
        "turkmen",
        "nynorsk",
        "maltese",
        "sanskrit",
        "luxembourgish",
        "myanmar",
        "tibetan",
        "tagalog",
        "malagasy",
        "assamese",
        "tatar",
        "hawaiian",
        "lingala",
        "hausa",
        "bashkir",
        "javanese",
        "sundanese",
        "cantonese",
        "burmese",
        "valencian",
        "flemish",
        "haitian",
        "letzeburgesch",
        "pushto",
        "panjabi",
        "moldavian",
        "moldovan",
        "sinhalese",
        "castilian",
        "mandarin",
        None,
    ] = Field(default=None, examples=["english"])
    response_format: Literal["json", "text"] = Field(examples=["json"])


class Segment(BaseModel):
    text: str = Field(examples=["Hello World", "Thanks"])
    start: float = Field(examples=[0, 3])
    end: float = Field(examples=[2, 4])


class TranscriptionResponse(BaseModel):
    audio_path: str = Field(examples=["/data/audio.wav"])
    text: Union[None, str] = Field(default=None, examples=["Hello World"])
    language: Literal[
        "english",
        "chinese",
        "german",
        "spanish",
        "russian",
        "korean",
        "french",
        "japanese",
        "portuguese",
        "turkish",
        "polish",
        "catalan",
        "dutch",
        "arabic",
        "swedish",
        "italian",
        "indonesian",
        "hindi",
        "finnish",
        "vietnamese",
        "hebrew",
        "ukrainian",
        "greek",
        "malay",
        "czech",
        "romanian",
        "danish",
        "hungarian",
        "tamil",
        "norwegian",
        "thai",
        "urdu",
        "croatian",
        "bulgarian",
        "lithuanian",
        "latin",
        "maori",
        "malayalam",
        "welsh",
        "slovak",
        "telugu",
        "persian",
        "latvian",
        "bengali",
        "serbian",
        "azerbaijani",
        "slovenian",
        "kannada",
        "estonian",
        "macedonian",
        "breton",
        "basque",
        "icelandic",
        "armenian",
        "nepali",
        "mongolian",
        "bosnian",
        "kazakh",
        "albanian",
        "swahili",
        "galician",
        "marathi",
        "punjabi",
        "sinhala",
        "khmer",
        "shona",
        "yoruba",
        "somali",
        "afrikaans",
        "occitan",
        "georgian",
        "belarusian",
        "tajik",
        "sindhi",
        "gujarati",
        "amharic",
        "yiddish",
        "lao",
        "uzbek",
        "faroese",
        "haitian creole",
        "pashto",
        "turkmen",
        "nynorsk",
        "maltese",
        "sanskrit",
        "luxembourgish",
        "myanmar",
        "tibetan",
        "tagalog",
        "malagasy",
        "assamese",
        "tatar",
        "hawaiian",
        "lingala",
        "hausa",
        "bashkir",
        "javanese",
        "sundanese",
        "cantonese",
        "burmese",
        "valencian",
        "flemish",
        "haitian",
        "letzeburgesch",
        "pushto",
        "panjabi",
        "moldavian",
        "moldovan",
        "sinhalese",
        "castilian",
        "mandarin",
        None,
    ] = Field(default=None, examples=["english"])
    segments: Union[list[Segment], None] = Field(
        default=None,
        examples=[
            "{{'language': 'english', 'text': 'Hello World', 'start': '0', 'end':'2'}}"
        ],
    )
    task: str = Field(default="transcribe")


@dataclass
class SpeechRecognitionInferenceConfig:
    model_config = ConfigDict(protected_namespaces=())
    model_id: Optional[str] = "openai/whisper-tiny"
    revision: Optional[str] = None
    model_dir: Optional[str] = os.path.expanduser("~")
    host: str = os.getenv("SRI_HOST", "0.0.0.0")
    port: int = os.getenv("SRI_PORT", 8080)
    auth_token: str = os.getenv("SRI_TOKEN", None)
    hf_access_token: str = os.getenv("HF_TOKEN", None)


def build_app(
    model_id: str,
    revision: str,
    model_dir: str,
    auth_token: str | None,
    hf_access_token: str | None,
) -> FastAPI:
    app = FastAPI(title="Speech Recognition Inference")

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

    pipe = load_pipeline(model_dir, model_id, revision, hf_access_token=hf_access_token)

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

        if auth_token:
            if Token != auth_token:
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
    @limiter.limit(limit_value="5/10seconds")
    def health_check(request: Request) -> Response:
        return Response()

    return app


def load_pipeline(
    cache_dir: str,
    model_id: str,
    revision: Optional[str] = None,
    hf_access_token: Optional[str] = None,
    batch_size: Optional[int] = 1,
) -> Pipeline:
    """Load a pipeline.

    Args
        -
    """

    model_path = os.path.join(
        os.path.join(cache_dir, "models--" + model_id.replace("/", "--"))
    )

    logger.debug("The model directory is {}".format(model_path))
    if not os.path.isdir(model_path):
        raise FileNotFoundError("The model directory does not exist")

    snapshot_dir = os.path.join(model_path, "snapshots")

    if revision is not None:
        if not os.path.isdir(os.path.join(snapshot_dir, revision)):
            raise FileNotFoundError(f"The model revision {revision} does not exist.")
    else:
        revisions = list(
            filter(lambda x: not x.startswith("."), os.listdir(snapshot_dir))
        )
        if len(revisions) == 0:
            logger.warning(
                "No revision provided and none found. Fetching the most recent"
                " available model."
            )
            download_hf_models(
                [model_id], hf_access_token=hf_access_token, cache_dir=cache_dir
            )
            revisions = filter(
                lambda x: not x.startswith("."), os.listdir(snapshot_dir)
            )
            revision = revisions[0]
        elif len(revisions) == 1:
            logger.info("No revision provided. Using the most recent model available.")
            revision = revisions[0]
        else:
            logger.info("No revision provided. Using the most recent model available.")
            revision = get_latest_commit(model_id, revisions)
    revision_dir = os.path.join(snapshot_dir, revision)

    logger.info("The model directory is {}".format(model_path))

    logger.info(f"Loading model {model_id} ({revision})...")

    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            logger.info(
                "MPS not available because the current PyTorch install was not built"
                " with MPS enabled."
            )
        else:
            device = "mps"
    else:
        device = "cpu"

    torch_dtype = torch.float16 if device != "cpu" else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        revision_dir,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )

    model.to(device)

    processor = AutoProcessor.from_pretrained(revision_dir)

    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        batch_size=batch_size,
    )


def transcribe_audio_file(
    audio_file: str, pipe: AutoModelForSpeechSeq2Seq, language: Optional[str] = None
):
    """
    Run a Hugging Face transcription inference pipeline.

    Args
        - audio_file: an absolute path to an audio file.
        - pipe: a AutoModelForSpeechSeq2Seq pipeline.
        - language: the language of the audio. If not provided,  Whisper would
    autodetect the language
    """
    if language is not None:
        result = pipe(
            audio_file,
            return_timestamps=True,
            return_language=True,
            generate_kwargs={"language": language},
        )
    else:
        result = pipe(audio_file, return_timestamps=True, return_language=True)

    return result


cli = typer.Typer()


@cli.command()
def main(
    model_id: str = typer.Option("openai/whisper-tiny", help=""),
    revision: str | None = typer.Option(None, help=""),
    model_dir: str = typer.Option("~/.cache/huggingface/hub", help=""),
    host: str = typer.Option("localhost", help=""),
    port: int = typer.Option("8080", help=""),
    auth_token: str | None = typer.Option(None, help=""),
    hf_token: str | None = typer.Option(None, help=""),
):
    config = SpeechRecognitionInferenceConfig()

    app = build_app(
        model_id=model_id or config.model_id,
        revision=revision or config.revision,
        model_dir=model_dir or config.model_dir,
        auth_token=auth_token or config.auth_token,
        hf_access_token=hf_token or config.hf_access_token,
    )

    uvicorn.run(
        app,
        host=host or config.host,
        port=port or config.port,
    )
