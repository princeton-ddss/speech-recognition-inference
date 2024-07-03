import sys

sys.path.append("/Users/jf3375/PycharmProjects/AudiotoTextAPI/src")

import requests
from fastapi.encoders import jsonable_encoder

from src.data_types.inputs.transcription import ModelInputs, \
    TranscriptionInputs


model_inputs = ModelInputs(model_name="whisper", model_size="test")
transcription_inputs = TranscriptionInputs(file_name="sample_data_short.wav",
                                           language="english",
                                           response_format="text")
json_body_dict = jsonable_encoder({"model_inputs": model_inputs,
                  "transcription_inputs":transcription_inputs})

#Options in requests.post
#params is for query inputs
#data only takes json inputs
#json:could convert dictionary to json if it could be done

r = requests.post("http://127.0.0.1:8000/transcription", json=json_body_dict)
print(r.json())

json_body_dict['transcription_inputs']['response_format'] = 'verbose_json'
r = requests.post("http://127.0.0.1:8000/transcription", json=json_body_dict)
print(r.json())
