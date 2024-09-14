from typing import Optional
from pydantic import BaseModel, ConfigDict


class SpeechRecognitionInferenceConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: Optional[str] = "openai/whisper-tiny"
    revision_id: Optional[str] = None
    model_dir: Optional[str] = None
    audio_dir: Optional[str] = None
    port: int = 8000
    host: str = "0.0.0.0"
    reload: bool = False
