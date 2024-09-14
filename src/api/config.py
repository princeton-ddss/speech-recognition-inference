from typing import Optional
from dataclasses import dataclass
from pydantic import ConfigDict


@dataclass
class SpeechRecognitionInferenceConfig:
    model_config = ConfigDict(protected_namespaces=())
    model_id: Optional[str] = "openai/whisper-tiny"
    revision_id: Optional[str] = None
    model_dir: Optional[str] = None
    audio_dir: Optional[str] = None
    port: int = 8000
    host: str = "0.0.0.0"
    auto_reload: bool = False
