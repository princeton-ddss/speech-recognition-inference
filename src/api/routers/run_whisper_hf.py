from typing import Union

def whisper_transcription_hf(
        file: str,
        pipe,
        language: Union[str, None] = None
):
    '''
    Run OpenAI Whisper on HuggingFace to get transcription texts with timestamp
    adjustments
    - **file**: A path to the audio file
    - **pipe**: A whisper model pipeline
    - **language**: Language in the audio; If not provided, Whisper would
    autodetect the language
    '''
    if language:
        result = pipe(file, return_timestamps=True,
                           return_language=True,
                  generate_kwargs={"language": language})
    else:
        result = pipe(file, return_timestamps=True, return_language=True)
    return result

