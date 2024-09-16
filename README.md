# speech-recognition-inference
A Speech-to-Text API compatible HuggingFace models.

## Quickstart

### Requirements
1. `ffmpeg`

### Pip
git clone https://github.com/princeton-ddss/speech-recognition-inference.git
cd speech-recognition-inference
python -m venv venv
pip install --upgrade pip
pip install . # or pip install -e . for development
python src/api/main.py \
  --port 8000:8000 \
  --model_id openai/whisper-large-v3 \
  --audio_dir /tmp \
  --model_dir $HOME/.cache/huggingface/hub
  --allow_downloads False

curl localhost:8000/transcribe \  -X POST \
  -d '{"audio_file": "female.wav", "response_format": "json"}' \
  -H 'Content-Type: application/json'

### Docker
docker run \
   -p 8000:8000 \
   -v $HOME/data:/data/models \
   -v /tmp:/data/audio
   speech-recognition-inference:latest \
  --port 8000 \
  --model_id openai/whisper-large-v3 \
  --audio_dir /data/audio \
  --model_dir /data/models
  --allow_downloads False

curl localhost:8000/transcribe \  -X POST \
  -d '{"audio_file": "female.wav", "response_format": "json"}' \
  -H 'Content-Type: application/json'


## Detailed Usage

### Model Selection

### Environment Variables
SRI_MODEL_DIR
SRI_AUDIO_DIR
