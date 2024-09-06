from typing import Optional
from pydantic import BaseModel, ConfigDict


class SpeechRecognitionInferenceConfig(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=()
    )  # model_ is protect by default... yuck...
    model_dir: str = "/data/models"
    model_id: Optional[str] = "openai/whisper-tiny"
    revision_id: Optional[str] = None
    audio_dir: str = "/data/audio"
    port: int = 8000
    host: str = "0.0.0.0"
