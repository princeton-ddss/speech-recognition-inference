import os

from typing import Optional
from dataclasses import dataclass
from pydantic import ConfigDict


@dataclass
class SpeechRecognitionInferenceConfig:
    model_config = ConfigDict(protected_namespaces=())
    model_id: Optional[str] = "openai/whisper-tiny"
    revision_id: Optional[str] = None
    model_dir: Optional[str] = os.getenv(
        "model_dir", "/Users/jf3375/Desktop/asr_api/models/Whisper_hf"
    )
    audio_dir: Optional[str] = os.getenv(
        "audio_dir", "/Users/jf3375/Desktop/asr_api/data"
    )
    port: int = os.getenv("model_dir", 8080)
    host: str = os.getenv("model_dir", "0.0.0.0")
    auto_reload: bool = False
