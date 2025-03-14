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

### API
Now start an API from the command-line:
```shell
speech_recognition_launcher \
  --port 8000:8000 \
  --model_id openai/whisper-tiny \
  --model_dir $HOME/.cache/huggingface/hub
```

We also provide a Docker image to start an API, 
`princetonddss/speech-recognition-inference`. To use it,
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

Once the application startup is complete, submit requests using any HTTP request
library or tool, e.g.,
```shell
curl localhost:8000/transcribe \  -X POST \
  -d '{"audio_file": "/tmp/female.wav", "response_format": "json"}' \
  -H 'Content-Type: application/json'
```

### Batch Processing
To run batch processing pipeline from the command-line:
```shell
speech_recognition_launcher \
  --batch_processing \
  --model_id openai/whisper-tiny \
  --model_dir $HOME/.cache/huggingface/hub
  --input_dir <input_dir_of_audio_files>
  --output_dir <output_dir_to_save_results>
```

Please only put all the audio files in the input directory. The 
transcription results of each audio file will be saved in csv files in the 
output directory.

The batch processing pipeline will by default chunk audio files into 30 
seconds chunks, calculate the batch size for batch processing, and run the 
batch processing pipeline on not procesesd input files which do not have 
transcription results in the output directory.

To manually change these default options:
```shell
speech_recognition_launcher \
  --batch_processing \
  --model_id openai/whisper-tiny \
  --model_dir $HOME/.cache/huggingface/hub
  --input_dir <input_dir_of_audio_files>
  --output_dir <output_dir_to_save_results>
  --input_chunks_dir <input_dir_of_audio_chunks>
  --chunking False
  --batch_size <batch_size>
  --rerun True
```
We also provide a Docker image to run batch processing pipeline, 
`princetonddss/speech-recognition-inference`. To use it,
simply run
```shell
docker run \
   -v $HOME/.cache/huggingface/hub:/data/models \
   -v <input_dir_of_audio_files>:/data/audio \
   -v <output_dir_to_save_results>:/outputs \
   speech-recognition-inference:latest \
   --batch_processing \
   --model_id openai/whisper-large-v3 \
   --model_dir /data/models \
   --input_dir /data/audio \
   --output_dir /outputs
```

## Detailed Usage

### Environment Variables
In addition to specifying settings at the command line, some settings can be provided through environment variables. These settings are:
- SRI_MODEL_DIR: the directory of model files
- SRI_HOST: the host of automatic speech recognition API
- SRI_PORT: the port of automatic speech recognition API
- HF_TOKEN: HuggingFace Access Token
- SRI_INPUT_DIR: input directory of audio files for batch processing
- SRI_OUTPUT_DIR: output directory to save transcription results


Command line arguments always take precedence over environment variables.
