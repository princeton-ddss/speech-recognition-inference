import os

from api.config import SpeechRecognitionInferenceConfig
from api.parser import SpeechRecognitionInferenceParser
from api.models import TranscriptionRequest, TranscriptionResponse, Segment
from api.pipeline import load_pipeline, transcribe_audio_file
from api.logger import logger

args = SpeechRecognitionInferenceParser.parse_args()

config = SpeechRecognitionInferenceConfig()

if args.model_dir is not None:
    config.model_dir = args.model_dir

if config.model_dir is None:
    raise Exception(
        "A model directory is required. Either set the SRI_AUDIO_DIR environment"
        " variables or pass in a value for --model_dir"
    )
if not os.path.isdir(config.model_dir):
    raise Exception(f"model_dir {config.model_dir} does not exist. ")

if args.audio_dir is not None:
    config.audio_dir = args.audio_dir

if config.audio_dir is None:
    raise Exception(
        "An audio directory is required. Either set the SRI_AUDIO_DIR environment"
        " variables or pass in a value for --audio_dir"
    )
if not os.path.isdir(config.audio_dir):
    raise Exception(f"audio_dir {config.audio_dir} does not exist. ")

if args.port is not None:
    config.port = args.port

if args.host is not None:
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

logger.info(f"Finished Loading model")
