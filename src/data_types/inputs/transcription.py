from pydantic import BaseModel, Field
from typing import Union, Literal


class ModelInputs(BaseModel):
    model_name: Literal["whisper", "seamless", "canary"] = Field(
        default = "whisper", examples = ["whisper"])
    model_size: Literal["test", "large"] = Field(default = "test",
                                                 examples=["test"])


class TranscriptionInputs(BaseModel):
    file_name: str = Field(examples = ["sample_data_short.wav"])
    language: Union[str, None] = Field(default = None, examples = ["english"])
    response_format: Literal["json", "text"] = Field(examples = [
        "json"])