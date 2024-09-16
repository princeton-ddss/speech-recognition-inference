import argparse

SpeechRecognitionInferenceParser = argparse.ArgumentParser()
SpeechRecognitionInferenceParser.add_argument("--model_id", help="The model to use, e.g., openai/whisper-tiny")
SpeechRecognitionInferenceParser.add_argument("--revision_id", help="The model revision to use, e.g., b5d...")
SpeechRecognitionInferenceParser.add_argument(
    "--model_dir",
    help="The location to look for model files, e.g., ~/.cache/huggingface/hub",
)
SpeechRecognitionInferenceParser.add_argument(
    "--audio_dir", help="The location to look for audio files, e.g., /tmp"
)
SpeechRecognitionInferenceParser.add_argument(
    "--port", type=int, help="The port to serve the API on (default: 8000)."
)
SpeechRecognitionInferenceParser.add_argument("--host",
                                              help="The host to serve the "
                                                   "API on (default: "
                                                   "0.0.0.0).")
SpeechRecognitionInferenceParser.add_argument(
    "--reload",
    type=bool,
    help="Automatically reload after changes to the source code (for development).",
    default=False,
)