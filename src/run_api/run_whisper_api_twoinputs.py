import sys

sys.path.append("/Users/jf3375/PycharmProjects/AudiotoTextAPI/src")

import requests
from fastapi.encoders import jsonable_encoder

from src.data_types.inputs.transcription import ModelInputs, \
    TranscriptionInputs

#
transcription_inputs = TranscriptionInputs(file_name="sample_data_short.wav",
                                           language="english",
                                           response_format="text")
# json_body_dict = jsonable_encoder({"transcription_inputs":transcription_inputs})
model_inputs = ModelInputs(model_name="whisper", model_size="test")
json_body_dict = jsonable_encoder({"model_inputs": model_inputs,
                  "transcription_inputs":transcription_inputs})

#Options in requests.post
#params is for query inputs
#data only takes json inputs
#json:could convert dictionary to json if it could be done

r = requests.post("http://0.0.0.0:8000/transcribe", json=json_body_dict)
print(r)
print(r.json())


json_body_dict['transcription_inputs']['response_format'] = 'json'
r = requests.post("http://127.0.0.1:8000/transcribe", json=json_body_dict)
print(r.json())