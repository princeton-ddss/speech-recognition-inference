import os
from typing import Optional
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, Pipeline, pipeline
from hf import download_hf_models, get_latest_commit
from logger import logger


def load_pipeline(
    cache_dir: str,
    model_id: str,
    revision: Optional[str] = None,
    hf_access_token: Optional[str] = None,
    batch_size: Optional[int] = 1,
) -> Pipeline:
    """Load a pipeline.

    Args
        -
    """

    model_path = os.path.join(
        os.path.join(cache_dir, "models--" + model_id.replace("/", "--"))
    )

    print("The model directory is {}".format(model_path))
    if not os.path.isdir(model_path):
        raise FileNotFoundError("The model directory does not exist")

    snapshot_dir = os.path.join(model_path, "snapshots")

    if revision is not None:
        if not os.path.isdir(os.path.join(snapshot_dir, revision)):
            raise FileNotFoundError(f"The model revision {revision} does not exist.")
    else:
        revisions = list(
            filter(lambda x: not x.startswith("."), os.listdir(snapshot_dir))
        )
        if len(revisions) == 0:
            logger.warning(
                "No revision provided and none found. Fetching the most recent"
                " available model."
            )
            download_hf_models(
                [model_id], hf_access_token=hf_access_token, cache_dir=cache_dir
            )
            revisions = filter(
                lambda x: not x.startswith("."), os.listdir(snapshot_dir)
            )
            revision = revisions[0]
        elif len(revisions) == 1:
            logger.info("No revision provided. Using the most recent model available.")
            revision = revisions[0]
        else:
            logger.info("No revision provided. Using the most recent model available.")
            revision = get_latest_commit(model_id, revisions)
    revision_dir = os.path.join(snapshot_dir, revision)

    logger.info("The model directory is {}".format(model_path))

    logger.info(f"Loading model {model_id} ({revision})...")

    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            logger.info(
                "MPS not available because the current PyTorch install was not built"
                " with MPS enabled."
            )
        else:
            device = "mps"
    else:
        device = "cpu"

    torch_dtype = torch.float16 if device != "cpu" else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        revision_dir,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )

    model.to(device)

    processor = AutoProcessor.from_pretrained(revision_dir)

    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        batch_size=batch_size,
    )


def transcribe_audio_file(
    audio_file: str, pipe: AutoModelForSpeechSeq2Seq, language: Optional[str] = None
):
    """
    Run a Hugging Face transcription inference pipeline.

    Args
        - audio_file: an absolute path to an audio file.
        - pipe: a AutoModelForSpeechSeq2Seq pipeline.
        - language: the language of the audio. If not provided,  Whisper would
    autodetect the language
    """
    if language is not None:
        result = pipe(
            audio_file,
            return_timestamps=True,
            return_language=True,
            generate_kwargs={"language": language},
        )
    else:
        result = pipe(audio_file, return_timestamps=True, return_language=True)

    return result
