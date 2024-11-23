import os

from typing import Optional
from dataclasses import dataclass
from pydantic import ConfigDict


@dataclass
class SpeechRecognitionInferenceConfig:
    model_config = ConfigDict(protected_namespaces=())
    model_id: Optional[str] = "openai/whisper-tiny"
    revision: Optional[str] = None
    model_dir: Optional[str] = os.getenv("SRI_MODEL_DIR",
                                         "/Users/jf3375/.blackfish/models")
    port: int = os.getenv("SRI_PORT", 8080)
    host: str = os.getenv("SRI_HOST", "0.0.0.0")
    auto_reload: bool = False
