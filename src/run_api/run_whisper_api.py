import sys

sys.path.append("../..")

import requests
from fastapi.encoders import jsonable_encoder

from src.data_types.inputs.transcription import ModelInputs, \
    TranscriptionInputs

#
transcription_inputs = {"file_name":"sample_data_short.wav",
                                           "language":"english",
                                           "response_format":"text"}
json_body_dict = jsonable_encoder(transcription_inputs)

#Options in requests.post
#params is for query inputs
#data only takes json inputs
#json:could convert dictionary to json if it could be done

r = requests.post("http://0.0.0.0:8000/transcribe", json=json_body_dict)
print(r)
print(r.json())


json_body_dict['response_format'] = 'json'
r = requests.post("http://127.0.0.1:8000/transcribe", json=json_body_dict)
print(r.json())
