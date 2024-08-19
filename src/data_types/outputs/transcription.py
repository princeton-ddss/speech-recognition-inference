from pydantic import BaseModel, Field
from typing import Union

class Segment(BaseModel):
    language: str = Field(examples=["english"])
    text: str = Field(examples=["Hello World", "Thanks"])
    start: float = Field(examples= [0, 3])
    end: float = Field(examples=[2, 4])


class TranscriptionOutputs(BaseModel):
    file: str = Field(examples=["sample_data_short.wav"])
    text: Union[None, str] = Field(default=None, examples=["Hello World"])
    segments: Union[list[Segment], None] = Field(default=None, examples=[
        "{{'text': 'Hello World', 'start': '0', 'end':'2'}}"])
    task: str = Field(default = "transcribe")
    used_model: str = Field(default=None, examples=["whisper_test"])
