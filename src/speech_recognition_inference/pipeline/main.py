import os
import re
from warnings import deprecated
from typing import Any, Optional, Self

from pydantic import BaseModel
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import numpy as np

from datasets import Audio
from speech_recognition_inference.utils import download_hf_models, get_latest_commit
from speech_recognition_inference.logger import logger


class BatchIterator:
    """Constructs batches of 30s audio segments from Audio."""

    def __init__(self, audio: Audio, batch_size: int = 1) -> Self:
        self.array = audio["array"]
        self.sampling_rate = audio["sampling_rate"]
        self.path = audio["path"]
        self.batch_size = batch_size
        self._idx = 0

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> list[np.array]:
        if self._idx < len(self.array):
            batch = []
            while len(batch) < self.batch_size and self._idx < len(self.array):
                sample = self.array[self._idx : self._idx + self.sample_len]
                self._idx += self.sample_len
                batch.append(sample)
            return batch
        else:
            raise StopIteration

    @property
    def sample_len(self) -> int:
        """Length of a 30s sample."""
        return self.sampling_rate * 30


@deprecated
def estimate_batch_size(device: str, allocation: float = 0.9):
    """Estimate the size of a batch based on allocation of a single 30s segment.

    Empirically, each 30s segment allocates ~0.001 GB. Thus, we should be able to process about
    1000 segments per GB of available memory.
    """

    if ("cuda" not in device) and (device != "mps"):
        raise ValueError(
            f"Unsupported device {device}. Supported device types are 'cuda' and 'mps'."
        )

    device_memory = (
        torch.cuda.get_device_properties(0).total_memory
        if "cuda" in device
        else torch.mps.recommended_max_memory()
    )
    logger.debug(f"Total device memory: {device_memory // 1e9} GB")

    allocated_memory = (
        torch.cuda.memory_allocated(0)
        if "cuda" in device
        else torch.mps.current_allocated_memory()
    )
    logger.debug(f"Allocated device memory: {allocated_memory // 1e9} GB")

    cached_memory = torch.cuda.memory_reserved(0) if "cuda" in device else 0
    logger.debug(f"Cached device memory: {cached_memory // 1e9} GB")

    free_memory = device_memory - (allocated_memory + cached_memory)
    available_memory = free_memory * allocation
    logger.debug(
        f"Available device memory: {available_memory // 1e9} of {free_memory // 1e9} GB"
    )

    batch_size = int(
        available_memory // 972_000
    )  # 972_000 = 80 x 3000 x 4 + 1 x 3000 x 4
    logger.debug(f"Estimated batch size: {batch_size}")

    if batch_size < 1:
        raise Exception(
            "Available GPU memory insufficient for the requested maximum file size."
        )

    return batch_size


def load_model(
    model_dir: str,
    model_id: str,
    revision: Optional[str] = None,
    device: Optional[str] = None,
    hf_access_token: Optional[str] = None,
) -> tuple[
    WhisperForConditionalGeneration,
    tuple[WhisperProcessor, dict[str, Any]] | WhisperProcessor,
]:
    logger.info(f"Loading model {model_id} from {model_dir}.")

    model_path = os.path.join(
        os.path.join(model_dir, "models--" + model_id.replace("/", "--"))
    )
    if not os.path.isdir(os.path.expanduser(model_path)):
        raise FileNotFoundError(
            "The model directory {} does not exist".format(model_path)
        )

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
                [model_id], hf_access_token=hf_access_token, cache_dir=model_dir
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

    logger.debug("The model path is {}".format(revision_dir))
    model = WhisperForConditionalGeneration.from_pretrained(revision_dir)
    processor = WhisperProcessor.from_pretrained(revision_dir)

    if not device:
        logger.info("No device selected. Checking for a default.")
        if torch.cuda.is_available():
            logger.info("CUDA is available. Moving model to cuda:0.")
            device = "cuda:0"
            model.to(device)
            torch.cuda.synchronize()  # ensures model is on device *before* calculating batch size!
        elif torch.mps.is_available():
            logger.info("MPS is available. Moving model to mps.")
            device = "mps"
            model.to(device)
            torch.mps.synchronize()  # ensures model is on device *before* calculating batch size!
        else:
            logger.info("CUDA is unavailable. Moving model to cpu.")
            device = "cpu"
            model.to(device)
    else:
        logger.info("Moving model to selected device.")
        model.to(device)

    return model, processor, device


class TranscriptChunk(BaseModel):
    text: Optional[str] = None
    timestamp: tuple[float, float]


class Transcription(BaseModel):
    language: Optional[str] = None
    text: str
    chunks: list[TranscriptChunk]

    @classmethod
    def from_string(cls, string: str, language: str, offset: int = 0) -> Self:
        """
        Construct a transcription from a raw result string.

        Args
            language (str): The language used for decoding.
            offset (int, optional): The value to add to the start and end of each
                timestamp. Defaults to 0.
        """
        pattern = re.compile(r"<\|(\d+\.\d+)\|>([^<]+)<\|(\d+\.\d+)\|>")
        matches = pattern.findall(string)

        if not matches:
            return Transcription(
                language=language,
                text=re.sub(r"<\|.*?\|> |!\s", "", string).strip(),
                chunks=[
                    {"timestamp": (0, 30), "text": string}
                ],  # No segments => remove random timestamps
            ).adjust_timestamps(offset=offset)
        else:
            return Transcription(
                language=language,
                text="".join([match[1] for match in matches]),
                chunks=[
                    {"timestamp": (float(match[0]), float(match[2])), "text": match[1]}
                    for match in matches
                ],
            ).adjust_timestamps(offset=offset)

    @classmethod
    def from_list(cls, transcripts: list[Self], language: str, offset: int = 0) -> Self:
        text = "".join([t.text for t in transcripts])
        chunks = [c for chunks in [t.chunks for t in transcripts] for c in chunks]
        return Transcription(
            language=language, text=text, chunks=chunks
        ).adjust_timestamps(offset=offset)

    def append(self, transcript: Self, offset: int = 0) -> Self:
        if offset > 0:
            transcript.adjust_timestamps(offset=offset)
        self.text += transcript.text
        self.chunks += transcript.chunks
        return self

    def adjust_timestamps(self, offset: int = 0) -> Self:
        """
        Adjusts the timestamps of all transcript chunks by applying an offset.

        Args:
            offset (int, optional): The value to add to the start and end of each
                timestamp. Defaults to 0.
        Returns:
            Self: The updated instance of the class with adjusted timestamps.
        """

        self.chunks = [
            TranscriptChunk(
                text=c.text,
                timestamp=(
                    offset + c.timestamp[0],
                    offset + c.timestamp[1],
                ),
            )
            for c in self.chunks
        ]
        return self


def process_batch(
    batch: list[np.array],
    model: WhisperForConditionalGeneration,
    processor: tuple[WhisperProcessor, dict[str, Any]] | WhisperProcessor,
    device: str,
    language: Optional[str] = None,
    sampling_rate: int = 16000,
    base_offset: int = 0,
) -> Transcription:
    """Run inference on a single batch of audio files.

    Args:
        model:
        audio_paths: the list of path to audio chunks files for batch processing
        output_dir: save transcription results in csv file in output path

    Returns:
        Transcription

    Raises:
        None
    """

    if language:
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=language, task="transcribe"
        )
    else:
        forced_decoder_ids = None

    input_features = processor(
        batch,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        return_attention_mask=True,
        predict_timestamps=True,
        device=device,
    ).to(device)

    predicted_ids = model.generate(
        input_features=input_features.input_features,
        attention_mask=input_features.attention_mask,
        return_timestamps=True,
        forced_decoder_ids=forced_decoder_ids,
    )

    if not language:
        language = processor.decode(predicted_ids[0, 1], device=device)[2:-2]

    results = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True,
        decode_with_timestamps=True,
        device=device,
    )

    return Transcription.from_list(
        [
            Transcription.from_string(
                string, language=language, offset=base_offset + 30 * idx
            )
            for idx, string in enumerate(results)
        ],
        language=language,
    )
