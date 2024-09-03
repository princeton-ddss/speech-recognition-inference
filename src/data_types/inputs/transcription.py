from pydantic import BaseModel, Field
from typing import Union, Literal

class TranscriptionInputs(BaseModel):
    file_name: str = Field(examples = ["sample_data_short.wav"])
    language: Union[str, None] = Field(default = None, examples = ["english"])
    response_format: Literal["json", "text"] = Field(examples = [
        "json"])