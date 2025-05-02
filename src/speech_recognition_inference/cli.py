import typer
from typing import Optional

import uvicorn


app = typer.Typer()


@app.command()
def launch(
    model_id: str = typer.Option("openai/whisper-tiny", help=""),
    revision: str | None = typer.Option(None, help=""),
    model_dir: str = typer.Option("~/.cache/huggingface/hub", help=""),
    host: str = typer.Option("localhost", help=""),
    port: int = typer.Option("8080", help=""),
    auth_token: str | None = typer.Option(None, help=""),
    hf_token: str | None = typer.Option(None, help=""),
):
    """Launch a speech-to-text API.

    The API performs transcription on individual audio files specified by path on the API server.
    """
    from speech_recognition_inference.api import (
        SpeechRecognitionInferenceConfig,
        build_app,
    )

    config = SpeechRecognitionInferenceConfig()

    app = build_app(
        model_id=model_id or config.model_id,
        revision=revision or config.revision,
        model_dir=model_dir or config.model_dir,
        auth_token=auth_token or config.auth_token,
        hf_access_token=hf_token or config.hf_access_token,
    )

    uvicorn.run(
        app,
        host=host or config.host,
        port=port or config.port,
    )


@app.command()
def pipeline(
    input_dir: str,
    model_id: str = typer.Option(
        "openai/whisper-tiny", help="A model ID to use for inference."
    ),
    model_dir: str = typer.Option(
        "~/.cache/huggingface/hub", help="A directory where model files are stored."
    ),
    revision: Optional[str] = typer.Option(
        None, help="The model revision to use for inference."
    ),
    batch_size: Optional[int] = typer.Option(
        1, help="The batch size to use for inference."
    ),
    allocation: Optional[float] = typer.Option(
        0.9, help="The proportion of available GPU memory to use."
    ),
    hf_access_token: Optional[str] = typer.Option(
        None, help="A Hugging Face access token to use."
    ),
    device: Optional[str] = typer.Option(None, help="The device to use."),
    language: Optional[str] = typer.Option(
        None, help="A language to use for transcription. Auto-detected per file if None"
    ),
    sampling_rate: int = typer.Option(16000, help="The audio sampling rate."),
    output_dir: Optional[str] = typer.Option(
        None, help="A location to store transcripts."
    ),
    rerun: bool = typer.Option(
        False, help="Re-run inference on *all* files in `input_dir`."
    ),
) -> None:
    """Perform batch speech-to-text inference.

    Inference is run for all audio files in `input_dir`. The command attempts to re-use previous transcription results, if possible.
    """

    import os
    import time
    import json
    from dotenv import load_dotenv

    from datasets import Dataset, Audio

    from speech_recognition_inference.pipeline import (
        BatchIterator,
        process_batch,
        load_model,
        Transcription,
    )
    from speech_recognition_inference.logger import logger

    if hf_access_token is None:
        load_dotenv()
        hf_access_token = os.getenv("HF_ACCESS_TOKEN", None)

    logger.info("Running speech recognition batch inference pipeline...")

    model_dir = os.path.abspath(model_dir)

    if output_dir is None:
        output_dir = os.path.join(input_dir, "transcripts")
        if not os.path.isdir(output_dir):
            logger.debug(f"Making new output directory {output_dir}")
            os.mkdir(output_dir)

    input_files = [
        f
        for f in os.listdir(input_dir)
        if not f.startswith(".") and os.path.isfile(os.path.join(input_dir, f))
    ]

    if rerun:
        paths = [os.path.join(input_dir, f) for f in input_files]
        dataset = Dataset.from_dict({"audio": paths}).cast_column(
            "audio", Audio(sampling_rate=sampling_rate)
        )
    else:
        logger.info("Determining audio files remaining to process...")
        output_files = [f for f in os.listdir(output_dir)]
        paths = [
            os.path.join(input_dir, f)
            for f in input_files
            if f"{os.path.splitext(f)[0]}.json" not in output_files
        ]
        if paths == []:
            logger.info("All audio files have already been processed!")
            return None
        else:
            logger.info(f"Found {len(paths)} remaining files.")
            dataset = Dataset.from_dict({"audio": paths}).cast_column(
                "audio", Audio(sampling_rate=sampling_rate)
            )

    model, processor, device = load_model(
        model_dir=model_dir,
        model_id=model_id,
        revision=revision,
        device=device,
        hf_access_token=hf_access_token,
    )

    starttime = time.time()
    step = 0
    nchunks = 0

    logger.info("Starting batch inference...")
    for data in dataset:
        logger.debug(f"File: {data['audio']['path']}")
        offset = 0  # seconds
        transcription = Transcription(language=language, text="", chunks=[])
        for batch in BatchIterator(data["audio"], batch_size=batch_size):
            tmp = process_batch(
                batch,
                model,
                processor,
                device,
                language,
                sampling_rate,
                base_offset=offset,
            )
            transcription.append(tmp)
            step += 1
            offset += 30 * len(batch)
            nchunks += len(batch)
            elapsed = time.time() - starttime
            logger.info(
                f"step: {step:4d}, nchunks: {nchunks:5d}, offset: {offset:5d}, elapsed:"
                f" {elapsed:0.3f} seconds"
            )

        base = os.path.basename(data["audio"]["path"])
        root, _ = os.path.splitext(base)
        with open(os.path.join(output_dir, f"{root}.json"), "w") as f:
            json.dump(transcription.model_dump(), f)

    logger.info(
        f"Finished processing {nchunks} chunks in"
        f" {(time.time() - starttime):0.3f} seconds."
    )

    logger.info("Done.")

    return None
