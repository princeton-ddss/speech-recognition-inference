import sys

sys.path.append("/Users/jf3375/PycharmProjects/AudiotoTextAPI/src")

# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=5678, stdoutToServer=True, stderrToServer=True)

import os
from fastapi import FastAPI
import uvicorn

from src.api.routers.run_whisper import whisper_transcription
from src.data_types.inputs.transcription import TranscriptionInputs
from src.data_types.outputs.transcription import Segment, TranscriptionOutputs

from src.api.routers.load_model import whisper_model, model_name, model_size, \
    input_folder



app = FastAPI()

#Take $MODEL_DIR

@app.get('/')
def get_main_page():
    return "Welcome to Audio-to-Text API Page"


@app.post("/transcribe", tags = ["transcription"],
          response_description="Transcription Outputs")
def run_transcription(
    transcription_inputs: TranscriptionInputs
) -> TranscriptionOutputs:
    """
    The main function to run the api to perform the speech-to-text
    transcription using the particular model
       - file_name: a audio file name uploaded to the default server address
       - language: language in the audio
       - response_format: choose json or text to return transcriptions
       with or without timestamps
    """

    #Get attributes from the instance
    file_name, language, response_format = transcription_inputs.file_name,\
    transcription_inputs.language, transcription_inputs.response_format

    file = os.path.join(input_folder, file_name)

    result = whisper_transcription(file, whisper_model, language)

    # Process outputs based on the response format
    output = TranscriptionOutputs(file=file_name, language=result['language'])

    output.text = result['text']
    if response_format == "json":
        segments = [None] * len(result["segments"])
        for idx, seg in enumerate(result["segments"]):
            segments[idx] = Segment(
                text=seg["text"], start=seg["start"], end=seg["end"]
            )
            output.segments = segments
    output.used_model = model_name + "_" + model_size
    return output

#API Run Port 8000
#Run it in CLI:
#uvicorn src.api.main:app --reload: need to start from the top level src
# folder to call other submodules
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload = True)
