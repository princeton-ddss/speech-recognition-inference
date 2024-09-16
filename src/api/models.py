from pydantic import BaseModel, Field
from typing import Union, Literal, Optional


class TranscriptionRequest(BaseModel):
    audio_path: str = Field(examples=[
        "/Users/jf3375/Desktop/asr_api/data/audio.wav"])
    language: Literal[
        "english",
        "chinese",
        "german",
        "spanish",
        "russian",
        "korean",
        "french",
        "japanese",
        "portuguese",
        "turkish",
        "polish",
        "catalan",
        "dutch",
        "arabic",
        "swedish",
        "italian",
        "indonesian",
        "hindi",
        "finnish",
        "vietnamese",
        "hebrew",
        "ukrainian",
        "greek",
        "malay",
        "czech",
        "romanian",
        "danish",
        "hungarian",
        "tamil",
        "norwegian",
        "thai",
        "urdu",
        "croatian",
        "bulgarian",
        "lithuanian",
        "latin",
        "maori",
        "malayalam",
        "welsh",
        "slovak",
        "telugu",
        "persian",
        "latvian",
        "bengali",
        "serbian",
        "azerbaijani",
        "slovenian",
        "kannada",
        "estonian",
        "macedonian",
        "breton",
        "basque",
        "icelandic",
        "armenian",
        "nepali",
        "mongolian",
        "bosnian",
        "kazakh",
        "albanian",
        "swahili",
        "galician",
        "marathi",
        "punjabi",
        "sinhala",
        "khmer",
        "shona",
        "yoruba",
        "somali",
        "afrikaans",
        "occitan",
        "georgian",
        "belarusian",
        "tajik",
        "sindhi",
        "gujarati",
        "amharic",
        "yiddish",
        "lao",
        "uzbek",
        "faroese",
        "haitian creole",
        "pashto",
        "turkmen",
        "nynorsk",
        "maltese",
        "sanskrit",
        "luxembourgish",
        "myanmar",
        "tibetan",
        "tagalog",
        "malagasy",
        "assamese",
        "tatar",
        "hawaiian",
        "lingala",
        "hausa",
        "bashkir",
        "javanese",
        "sundanese",
        "cantonese",
        "burmese",
        "valencian",
        "flemish",
        "haitian",
        "letzeburgesch",
        "pushto",
        "panjabi",
        "moldavian",
        "moldovan",
        "sinhalese",
        "castilian",
        "mandarin",
        None,
    ] = Field(default=None, examples=["english"])
    response_format: Literal["json", "text"] = Field(examples=["json"])


class Segment(BaseModel):
    language: Optional[str] = Field(examples=["english"])
    text: str = Field(examples=["Hello World", "Thanks"])
    start: float = Field(examples=[0, 3])
    end: float = Field(examples=[2, 4])


class TranscriptionResponse(BaseModel):
    audio_path: str = Field(examples=[
        "/Users/jf3375/Desktop/asr_api/data/audio.wav"])
    text: Union[None, str] = Field(default=None, examples=["Hello World"])
    segments: Union[list[Segment], None] = Field(
        default=None,
        examples=[
            "{{'language': 'english', 'text': 'Hello World', 'start': '0', 'end':'2'}}"
        ],
    )
    task: str = Field(default="transcribe")
