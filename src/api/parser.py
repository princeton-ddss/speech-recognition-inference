from argparse import ArgumentParser


class SpeechRecognitionInferenceParser(ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument(
            "--model_id", help="The model to use (default: openai/whisper-tiny)."
        )
        self.add_argument(
            "--revision_id",
            help=(
                "The model revision to use. Defaults to the most recent revision"
                " available."
            ),
        )
        self.add_argument(
            "--model_dir",
            help=(
                "The location to look for model files, e.g., ~/.cache/huggingface/hub."
            ),
        )
        self.add_argument(
            "--port", type=int, help="The port to serve the API on (default: 8080)."
        )
        self.add_argument(
            "--host", help="The host to serve the API on (default: 0.0.0.0)."
        )
        self.add_argument(
            "--reload",
            help=(
                "Automatically reload after changes to the source code (for"
                " development)."
            ),
            action="store_true",
        )
