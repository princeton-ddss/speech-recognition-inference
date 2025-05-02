# speech-recognition-inference
This repo provides a command-line tool for performing automatic speech-to-text tasks (i.e., "transcription") using open source models from Hugging Face Hub. For interactive tasks, it allows users to spin up an inference API. For bulk processing, it furnishes a pipeline for running inference on the contents of a specified directory.

## Quickstart

### Pip
First, [install](https://ffmpeg.org/download.html) `ffmpeg` if it is not already installed on your machine.

Next, clone the repository and install it:
```shell
git clone https://github.com/princeton-ddss/speech-recognition-inference.git
cd speech-recognition-inference
python -m venv .venv
pip install --upgrade pip
pip install . # or pip install -e . for development
```

To start an API from the command-line, simply run:
```shell
speech_recognition launch \
  --port 8000:8000 \
  --model-id openai/whisper-tiny \
  --model-dir $HOME/.cache/huggingface/hub
```

Once the application startup is complete, you can submit requests using any HTTP request
library or tool, e.g.,
```shell
curl localhost:8000/transcribe \
  -X POST \
  -d '{"audio_file": "/tmp/female.wav", "response_format": "json"}' \
  -H 'Content-Type: application/json'
```

To run batch processing, run:
```shell
speech_recognition pipeline /data/audio \
  --model-id openai/whisper-tiny \
  --model-dir $HOME/.cache/huggingface/hub
```

### Docker
We also provide a Docker image, `ghcr.io/princeton-ddss/speech-recognition-inference`. To run the API via Docker use the following command:
```shell
docker run \
  -p 8000:8000 \
  -v $HOME/.cache/huggingface/hub:/data/models \
  -v /tmp:/data/audio \
  ghcr.io/princeton-ddss/speech-recognition-inference:latest \
  launch \
  --port 8000 \
  --model_id openai/whisper-large-v3 \
  --model_dir /data/models
```

This command makes the API available on `localhost:8000` and makes host model and audio files available to the container via bind mounting. Sending requests is the same as above, but note that the container only has access to bind mounted files. Above, this means that requests should replace `/tmp` with `/data/audio`:

```shell
curl localhost:8000/transcribe \
  -X POST \
  -d '{"audio_file": "/data/audio/female.wav", "response_format": "json"}' \
  -H 'Content-Type: application/json'
```

To run batch processing via Docker, replace the `launch` command with `pipeline` and update the options:
```shell
docker run \
  -v $HOME/.cache/huggingface/hub:/data/models \
  -v /tmp:/data/audio \
  ghcr.io/princeton-ddss/speech-recognition-inference:latest \
  pipeline \
  /data/audio \
  --model_id openai/whisper-large-v3 \
  --model_dir /data/models
```

Again, note that `/tmp` is bound to `/data/audio` on the host, so this command runs inference on all audio files in `/tmp` on the host.


## Detailed Usage
Full usage details are available via the `--help` option. For example,
```shell
❯ speech_recognition --help

 Usage: speech_recognition [OPTIONS] COMMAND [ARGS]...

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --install-completion          Install completion for the current shell.                                                                          │
│ --show-completion             Show completion for the current shell, to copy it or customize the installation.                                   │
│ --help                        Show this message and exit.                                                                                        │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ launch     Launch a speech-to-text API.                                                                                                          │
│ pipeline   Perform batch speech-to-text inference.                                                                                               │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

### Environment Variables
In addition to specifying settings at the command line, some settings can be provided through environment variables. These settings are:

- `SRI_HOST` - Host to use for the API.
- `SRI_PORT` - Port to use for the API.
- `SRI_TOKEN` - Authentication token to use for authenticated API access.
- `HF_ACCESS_TOKEN` - Authentication token to use for authentication with the Hugging Face Hub API.

Users can set environment variables by `export`ing or via a `.env` file. *Command line arguments always take precedence over environment variables.*
