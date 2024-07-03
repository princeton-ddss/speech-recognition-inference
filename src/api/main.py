import sys

sys.path.append("/Users/jf3375/PycharmProjects/AudiotoTextAPI/src")

import os
from fastapi import FastAPI
import uvicorn


from src.api.routers.run_whisper import whisper_transcription
from src.data_types.inputs.transcription import ModelInputs, TranscriptionInputs
from src.data_types.outputs.transcription import Segment, TranscriptionOutputs


import os

# Load model and file inputs from docker run from users
# May need to do the volume mount
default_model_folder = "/Users/jf3375/Dropbox (Princeton)/models"
default_model_name = "Whisper"
default_model_size = "test"
default_input_folder = "/Users/jf3375/Dropbox (Princeton)/inputs"

model_folder = os.getenv("MODEL_FOLDER", default_model_folder)
model_name = os.getenv("MODEL_NAME", default_model_name)
model_size = os.getenv("MODEL_SIZE", default_model_size)
input_folder = os.getenv("INPUT_FOLDER", default_input_folder)

# Load model during API set up
if model_name == "whisper":
    if model_size == "test":
        model = os.path.join(model_folder, "Whisper", "tiny.pt")
    if model_size == "large":
        model = os.path.join(model_folder, "Whisper", "large-v2.pt")


app = FastAPI()

#Take $MODEL_DIR

@app.get('/')
def get_main_page():
    return "Welcome to Audio-to-Text API Page"


@app.post("/transcribe", tags = ["transcription"],
          response_description="Transcription Outputs")
def run_transcription(
    model_inputs: ModelInputs,
    transcription_inputs: TranscriptionInputs
) -> TranscriptionOutputs:
    """
    The main function to run the api to perform the speech-to-text
    transcription using the particular model
    - **model_inputs**:
       - model_name: the model name
       - model_size: the model size
    - **transcription_inputs**:
       - file_name: a audio file name uploaded to the default server address
       - language: language in the audio
       - response_format: choose json or text to return transcriptions
       with or without timestamps
    """

    #Get attributes from the instance
    model_name, model_size = model_inputs.model_name, model_inputs.model_size
    file_name, language, response_format = transcription_inputs.file_name,\
    transcription_inputs.language, transcription_inputs.response_format

    # Set directory model folder which mounts with model folder on cluster
    model_folder = "/Users/jf3375/Dropbox (Princeton)/models"

    # Set directory files folder which mounts with files folder on cluster
    input_folder = "/Users/jf3375/Dropbox (Princeton)/inputs"
    file = os.path.join(input_folder, file_name)

    if model_name == "whisper":
        if model_size == "test":
            model = os.path.join(model_folder, "Whisper", "tiny.pt")
        if model_size == "large":
            model = os.path.join(model_folder, "Whisper", "large-v2.pt")
        result = whisper_transcription(file, model, language)

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
