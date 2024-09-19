# speech-recognition-inference
This repo provides an API that serves speech-to-text models from HuggingFace Hub.
Simply choose the model and (optionally) revision that you would like to serve, and
`speech-recognition-inference` will launch an API that performs automatic speech
recognition on any audio file you specify.

## Quickstart

### Pip
First, install `ffmpeg` if this is not already installed on your machine.

Next, clone the repository and install:
```shell
git clone https://github.com/princeton-ddss/speech-recognition-inference.git
cd speech-recognition-inference
python -m venv venv
pip install --upgrade pip
pip install . # or pip install -e . for development
```

Now start an API from the command-line:
```shell
speech_recognition_launcher \
  --port 8000:8000 \
  --model_id openai/whisper-tiny \
  --model_dir $HOME/.cache/huggingface/hub
```

Once the application startup is complete, submit requests using any HTTP request
library or tool, e.g.,
```shell
curl localhost:8000/transcribe \  -X POST \
  -d '{"audio_file": "/tmp/female.wav", "response_format": "json"}' \
  -H 'Content-Type: application/json'
```

### Docker
We also provide a Docker image, `princetonddss/speech-recognition-inference`. To use it,
simply run
```shell
docker run \
   -p 8000:8000 \
   -v $HOME/.cache/huggingface/hub:/data/models \
   -v /tmp:/data/audio \
   speech-recognition-inference:latest \
  --port 8000 \
  --model_id openai/whisper-large-v3 \
  --model_dir /data/models
```

Sending requests works exactly as shown above.


## Detailed Usage

### Environment Variables
In addition to specifying settings at the command line, some settings can be provided through environment variables. These settings are:
- SRI_MODEL_DIR
- SRI_HOST
- SRI_PORT
- HF_ACCESS_TOKEN

Command line arguments always take precedence over environment variables.
