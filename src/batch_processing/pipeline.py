import os
from typing import Optional
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from batch_processing.logger import logger
from hf import download_hf_models, get_latest_commit
def load_model(cache_dir: str,
    model_id: str,
    revision: Optional[str] = None,
    hf_access_token: Optional[str] = None):
    #Load Model
    model_path = os.path.join(
        os.path.join(cache_dir, "models--" + model_id.replace("/", "--"))
    )
    if not os.path.isdir(model_path):
        raise FileNotFoundError("The model directory {} does not exist".format(model_path))

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

    logger.info("The model path is {}".format(revision_dir))
    model = WhisperForConditionalGeneration.from_pretrained(revision_dir)
    processor = WhisperProcessor.from_pretrained(revision_dir)
    return model, processor


