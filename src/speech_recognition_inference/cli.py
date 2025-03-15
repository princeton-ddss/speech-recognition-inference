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
        None, help="The batch size to use for inference."
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
    from dotenv import load_dotenv

    from speech_recognition_inference import pipeline
    from speech_recognition_inference.logger import logger

    if hf_access_token is None:
        load_dotenv()
        hf_access_token = os.getenv("HF_ACCESS_TOKEN", None)

    logger.info("Running speech recognition batch inference pipeline...")

    model_dir = os.path.abspath(model_dir)

    if output_dir is None:
        output_dir = os.path.join(input_dir, "out")
        if not os.path.isdir(output_dir):
            logger.debug(f"Making new output directory {output_dir}")
            os.mkdir(output_dir)

    if rerun:
        try:
            chunk_files = pipeline.chunk_all(input_dir)
        except Exception as e:
            logger.error(f"Failed to chunk input directory: {e}")
            return None
    else:
        if not os.path.isdir(os.path.join(input_dir, "chunks")):
            logger.info(f"No chunks found. Chunking all audio files in {input_dir}...")
            try:
                chunk_files = pipeline.chunk_all(input_dir)
            except Exception as e:
                logger.error(f"Failed to chunk input directory: {e}")
                return None
        else:
            logger.info(
                "Found existing chunks. Determining audio files remaining to process..."
            )
            input_files = [
                f
                for f in os.listdir(input_dir)
                if not f.startswith(".") and os.path.isfile(os.path.join(input_dir, f))
            ]
            output_files = [f for f in os.listdir(output_dir)]
            remaining_files = [
                f
                for f in input_files
                if f"{os.path.splitext(f)[0]}.json" not in output_files
            ]
            if remaining_files == []:
                logger.info("All audio files have already been processed!")
                return None
            else:
                logger.info(
                    f"Found {len(remaining_files)} remaining files. Attempting to"
                    " chunk..."
                )
                try:
                    chunk_files = pipeline.chunk_many(
                        [os.path.join(input_dir, f) for f in remaining_files],
                        output_dir=os.path.join(input_dir, "chunks"),
                    )
                except Exception as e:
                    logger.error(f"Failed to chunk remaining files: {e}")
                    return None
            if len(chunk_files) == 0:
                logger.info("All audio files have already been processed!")
                return None

    model, processor, device = pipeline.load_model(
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
    for batch in pipeline.BatchIterator(chunk_files, batch_size=batch_size):
        pipeline.process_batch(
            model, processor, batch, device, language, sampling_rate, output_dir
        )

        step += 1
        nchunks += len(batch)
        elapsed = time.time() - starttime
        logger.info(
            f"step: {step:4d}, nchunks: {nchunks:5d}, elapsed: {elapsed:0.3f} seconds"
        )

    logger.info(
        f"Finished processing {nchunks} chunks in"
        f" {(time.time() - starttime):0.3f} seconds."
    )

    logger.info(f"Concatenating results and writing to {output_dir}...")
    pipeline.concatenate_results(output_dir)

    logger.info("Cleaning up temporary outputs...")
    pipeline.clean_up(output_dir)

    logger.info("Done.")

    return None
