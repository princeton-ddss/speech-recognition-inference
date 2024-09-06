from pydantic import BaseModel, Field
from typing import Union, Literal


class TranscriptionRequest(BaseModel):
    audio_file: str = Field(examples=["audio.wav"])
    language: Union[str, None] = Field(default=None, examples=["english"])
    response_format: Literal["json", "text"] = Field(examples=["json"])


class Segment(BaseModel):
    language: str = Field(examples=["english"])
    text: str = Field(examples=["Hello World", "Thanks"])
    start: float = Field(examples=[0, 3])
    end: float = Field(examples=[2, 4])


class TranscriptionResponse(BaseModel):
    audio_file: str = Field(examples=["audio.wav"])
    text: Union[None, str] = Field(default=None, examples=["Hello World"])
    segments: Union[list[Segment], None] = Field(
        default=None,
        examples=[
            "{{'language': 'english', 'text': 'Hello World', 'start': '0', 'end':'2'}}"
        ],
    )
    task: str = Field(default="transcribe")
