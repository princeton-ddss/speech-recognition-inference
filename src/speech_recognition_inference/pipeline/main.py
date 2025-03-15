import os
import re
import json
from typing import Any, Optional

import shutil
from pydub import AudioSegment, utils
from pydub.exceptions import CouldntDecodeError
from datasets import Dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

from speech_recognition_inference.utils import download_hf_models, get_latest_commit
from speech_recognition_inference.logger import logger


class BatchIterator:
    def __init__(self, items: list, batch_size: int):
        self.items = items
        self.batch_size = batch_size
        self._idx = 0

    @property
    def size(self):
        return len(self.items)

    # @property
    # def batch_size(self):
    #     batch_size = calculate_batch_size(
    #         max_file_size=max_file_size,
    #         remaining_proportion=remaining_proportion,
    #         device=device,
    #     )

    def __iter__(self):
        return self

    def __next__(self) -> list:
        """Construct the next batch of items."""

        if self._idx < self.size:
            batch = self.items[self._idx : min(self._idx + self.batch_size, self.size)]
            self._idx += self.batch_size
            return batch
        else:
            raise StopIteration


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


def chunk_one(audio_path: str, output_dir: str, chunk_len: int = 30) -> list[str]:
    """Split a audio file in equal-length chunks and write to file.

    Copy the file to the output directory without chunking if the original file length is less than
    the desired chunk length.

    Args:
        audio_path: The path of an input audio file.
        output_dir: A directory to store output audio chunks.
        chunk_len: The desired chunk length in seconds.

    Returns:
        A list of paths to the newly created audio chunk files.

    Raises:
        None
    """

    base = os.path.basename(audio_path)
    logger.debug(f"Creating chunks for {base}")

    audio = AudioSegment.from_file(audio_path)

    chunk_paths = []
    root, ext = os.path.splitext(base)
    if len(audio) <= chunk_len * 1e3:
        logger.debug(f"Audio file {base} is less than 30s. Creating a single chunk.")
        chunk_path = os.path.join(output_dir, f"{root}_0{ext}")
        shutil.copy(
            audio_path,
        )
        chunk_paths.append(chunk_path)
    else:
        chunks = utils.make_chunks(audio, chunk_len * 1e3)
        for idx, chunk in enumerate(chunks):
            chunk_path = os.path.join(output_dir, f"{root}_{idx}{ext}")
            chunk.export(chunk_path)
            chunk_paths.append(chunk_path)

    return chunk_paths


def chunk_many(
    audio_paths: list[str], output_dir: str, chunk_len: int = 30
) -> list[str]:
    """Split a list of audio files into 30 seconds chunks.

    Args:
        audio_paths: A list of audio files to chunk.
        output_dir: A directory to store output audio chunks.
        chunk_len: The desired chunk length in seconds.

    Returns:
        None

    Raises:
        Exception: Audio chunks already present in `chunks` directory.
    """

    if not os.path.exists(output_dir):
        logger.debug(f"Making new chunks directory {output_dir}")
        os.makedirs(output_dir)

    logger.debug(f"Creating audio chunks and writing to {output_dir}.")

    audio_paths = [
        p
        for p in audio_paths
        if not os.path.basename(p).startswith(".") and os.path.isfile(p)
    ]

    output_files = []
    nsuccess = 0
    ntotal = len(audio_paths)
    for audio_path in audio_paths:
        try:
            chunks = chunk_one(audio_path, output_dir, chunk_len=chunk_len)
        except CouldntDecodeError:
            logger.warning(
                f"Could not decode input {os.path.basename(audio_path)}. Skipping."
            )
            continue

        output_files.extend(chunks)
        nsuccess += 1

    logger.debug(
        f"Chunking complete. Successfully chunked {nsuccess} of {ntotal} files."
    )

    return output_files


def chunk_all(
    audio_dir: str, output_dir: str | None = None, chunk_len: int = 30
) -> list[str]:
    """Split all audio files in a directory into 30 seconds chunks.

    Args:
        audio_dir: A directory containing audio files to chunk.
        output_dir: A directory to store output audio chunks. Default: <audio_dir>/chunks.
        chunk_len: The desired chunk length in seconds.

    Returns:
        None

    Raises:
        None
    """

    logger.debug(
        "Creating audio chunks and writing to"
        f" {os.path.abspath(os.path.join(audio_dir, 'chunks'))}"
    )

    output_dir = os.path.join(audio_dir, "chunks")
    if not os.path.exists(output_dir):
        logger.debug(f"Making new chunks directory {output_dir}")
        os.makedirs(output_dir)

    input_files = [
        f
        for f in os.listdir(audio_dir)
        if not f.startswith(".") and os.path.isfile(os.path.join(audio_dir, f))
    ]
    output_files = []
    nsuccess = 0
    ntotal = len(input_files)
    for f in input_files:
        try:
            chunks = chunk_one(
                os.path.join(audio_dir, f),
                output_dir,
                chunk_len=chunk_len,
            )
        except CouldntDecodeError:
            logger.warning(f"Could not decode input {f}. Skipping.")
            continue

        output_files.extend(chunks)
        nsuccess += 1

    logger.debug(
        f"Chunking complete. Successfully chunked {nsuccess} of {ntotal} files."
    )

    return output_files


def process_batch(
    model: WhisperForConditionalGeneration,
    processor: tuple[WhisperProcessor, dict[str, Any]] | WhisperProcessor,
    audio_paths: list[str],
    device: str,
    language: Optional[str] = None,
    sampling_rate: int = 16000,
    output_dir: Optional[str] = None,
) -> None:
    """Run inference on a single batch of audio files.

    Args:
        model:
        audio_paths: the list of path to audio chunks files for batch processing
        output_dir: save transcription results in csv file in output path

    Returns:
        None

    Raises:
        None
    """

    if language:
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=language, task="transcribe"
        )
    else:
        forced_decoder_ids = None

    dataset = Dataset.from_dict({"audio": audio_paths}).cast_column(
        "audio", Audio(sampling_rate=sampling_rate)
    )

    batch = [data["audio"]["array"] for data in dataset]

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

    transcriptions = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True,
        decode_with_timestamps=True,
        device=device,
    )

    for path, raw_string in zip(audio_paths, transcriptions):
        base = os.path.basename(path)
        root, _ = os.path.splitext(base)
        result = parse_string_to_result(raw_string, language=language)
        with open(os.path.join(output_dir, f"{root}.json"), "w") as f:
            json.dump(result, f)

    return None


def parse_string_to_result(input_string, language):
    """Parse raw string to result format."""

    pattern = re.compile(r"<\|(\d+\.\d+)\|>([^<]+)<\|(\d+\.\d+)\|>")
    matches = pattern.findall(input_string)

    result = {}
    result["language"] = language
    if not matches:
        # No segments => remove random timestamps
        result["text"] = re.sub(r"<\|.*?\|> |!\s", "", input_string).strip()
        result["chunks"] = [{"timestamp": (0, 30), "text": input_string}]
    else:
        result["text"] = ""
        result["chunks"] = []
        for match in matches:
            result["text"] += match[1]
            result["chunks"] += [
                {"timestamp": (float(match[0]), float(match[2])), "text": match[1]}
            ]

    return result


def concatenate_results(output_dir: str, chunk_len: int = 30) -> None:
    """Merge transcription results into a single file."""

    output_files = [c for c in os.listdir(output_dir) if c.endswith(".json")]
    output_files.sort()  # sort by timestamp

    results = {}
    for output_file in output_files:
        match = re.search(r"(.*)_(\d+)\.json$", output_file)
        if not match:
            continue  # skip previously concatenated results
        parent = match.group(1)
        index = int(match.group(2))
        with open(os.path.join(output_dir, output_file), "r") as f:
            result = json.load(f)

        if parent in results:
            results[parent]["text"] += result["text"]
            results[parent]["chunks"] += fix_timestamps(
                result["chunks"], index, chunk_len=chunk_len
            )
        else:
            if index > 0:
                raise Exception("First result should be order 0.")
            results[parent] = result

    for parent, result in results.items():
        with open(os.path.join(output_dir, f"{parent}.json"), "w") as f:
            json.dump(result, f)


def clean_up(output_dir: str) -> None:
    """Remove temporary output files."""
    output_files = os.listdir(output_dir)
    for f in output_files:
        match = re.search(r"(.*)_(\d+)\.json$", f)
        if match:
            os.remove(os.path.join(output_dir, f))


def fix_timestamps(chunks: list[dict], index: int, chunk_len: int = 30) -> list[dict]:
    offset = chunk_len * index
    for chunk in chunks:
        chunk["timestamp"] = (
            chunk["timestamp"][0] + offset,
            chunk["timestamp"][1] + offset,
        )

    return chunks
