import os
import torch
from datasets import Dataset, Audio
from typing import Optional
from batch_processing.parsing import parse_string_into_result, parse_results_to_csv
from batch_processing.pipeline import load_model
from batch_processing.chunking import chunking_dir, merge_chunks_results
from collections import deque
import warnings


def batch_processing(
    model,
    processor,
    audio_paths,
    device,
    language=None,
    sampling_rate=16000,
    output_dir=None,
):
    """
    audio_paths: the list of path to audio chunks files for batch processing
    output_dir: save transcription results in csv file in output path
    """
    audio_paths.sort()
    dataset = Dataset.from_dict({"audio": audio_paths}).cast_column("audio", Audio())
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))
    batch_size = len(dataset)
    batch = [None] * batch_size
    for idx, data in enumerate(dataset):
        batch[idx] = data["audio"]["array"]

    # Set language if exists
    if language:
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=language, task="transcribe"
        )
    else:
        forced_decoder_ids = None

    # Prepare input features
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

    transcriptions_batch = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True,
        decode_with_timestamps=True,
        device=device,
    )

    results_batch = [None] * batch_size
    for idx, result_string in enumerate(transcriptions_batch):
        results_batch[idx] = parse_string_into_result(result_string, language)
    if output_dir:
        for idx, result in enumerate(results_batch):
            output_name = os.path.basename(audio_paths[idx]).split(".")[0]
            parse_results_to_csv(result, output_dir, output_name)
    # Merge results of chunks of one file together
    merge_chunks_results(output_dir)
    return results_batch


def calculate_batch_size(max_file_size_mb=5, buffer_proportion=0.5, device=None):
    """
    Dynamically Calculate Batch Size in Real Time
    """
    # Dynamically calculate batch size based on available resources
    if not device:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if device == "cuda:0" or device == "cuda":
        # Get the total memory
        total_memory = torch.cuda.get_device_properties(0).total_memory
        # Get the allocated memory
        allocated_memory = torch.cuda.memory_allocated(0)
        # Get the cached memory
        cached_memory = torch.cuda.memory_reserved(0)
        # Calculate the free memory
        # Multiply the free memory by 80% to have enough buffer
        free_memory = (
            total_memory - (allocated_memory + cached_memory)
        ) * buffer_proportion
        free_memory_mb = free_memory / 1e6
    elif device == "mps":
        # Get the recommend max memory
        total_memory = torch.mps.recommended_max_memory()
        # Get the allocated memory, which includes cached memory
        used_memory = torch.mps.driver_allocated_memory()
        # Calculate the free memory
        # Multiply the free memory by 80% to have enough buffer
        free_memory = (total_memory - used_memory) * buffer_proportion
        free_memory_mb = free_memory / 1e6

    if device == "cpu":
        batch_size = 1
    else:
        batch_size = int(free_memory_mb // max_file_size_mb)
    return batch_size


def run_batch_processing_queue(
    cache_dir: str,
    model_id: str,
    input_dir: str,
    revision: Optional[str] = None,
    hf_access_token: Optional[str] = None,
    device=None,
    chunking=True,
    language=None,
    sampling_rate=16000,
    output_dir=None,
):
    """
    This function assumes that the length of each audio file is less than or
    equal to 20 minutes for Whisper to run properly under batch processing.
    output_dir: save transcription results in csv file in output path

    chunking: If True, chunk files into input_dir and save chunks in
    input_chunks_dir.
    If False, input_dir already contains all chunks
    """
    # Load Model
    model, processor = load_model(
        cache_dir=cache_dir,
        model_id=model_id,
        revision=revision,
        hf_access_token=hf_access_token,
    )

    if chunking:
        # Chunk files into input_dir and save chunks in input_chunks_dir
        input_dir = chunking_dir(input_dir, chunk_len_in_secs=30)
    else:
        warnings.warn(
            "Check to make sure all audio files in the directory                      "
            " are less than 30 seconds. The model would only                      "
            " transcribe first 30 seconds for each file"
        )
    # Create a queue for batch processing of chunks of audio files
    chunk_files = os.listdir(input_dir)
    chunk_files.sort()  # Make sure chunks of same files get processed together
    chunk_files_path = [
        os.path.join(input_dir, chunk_file)
        for chunk_file in chunk_files
        if not chunk_file.startswith(".")
    ]
    chunks_queue = deque(chunk_files_path)

    # Load Model on Device
    if not device:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Process audio chunk files under batch processing
    audio_paths = []
    while chunks_queue:
        batch_size = calculate_batch_size(
            max_file_size_mb=5, buffer_proportion=0.5, device=device
        )
        for _ in range(batch_size):
            if chunks_queue:
                audio_paths.append(chunks_queue.pop())
            else:
                break
        results_batch = batch_processing(
            model, processor, audio_paths, device, language, sampling_rate, output_dir
        )
    return results_batch


# Run on laptop
# input_dir = "/Users/jf3375/Desktop/asr_api/data/test"
# device = "mps"
# output_dir = "/Users/jf3375/Desktop/asr_api/output/localview_test"
# cache_dir = "/Users/jf3375/Princeton Dropbox/Junying Fang/asr_api/models/Whisper_hf"
# model_id = "openai/whisper-tiny"
#
# results = run_batch_processing_queue(
#     cache_dir=cache_dir,
#     model_id=model_id,
#     input_dir=input_dir,
#     device=device,
#     chunks=True,
#     language="en",
#     output_dir=output_dir,
# )
# print(results)

# Run on Della
input_dir="/scratch/gpfs/jf3375/asr_api/data/test"
device="cuda:0"
output_dir="/scratch/gpfs/jf3375/asr_api/output"
cache_dir = "/scratch/gpfs/jf3375/asr_api/models/Whisper_hf"
model_id = "openai/whisper-tiny"

results = run_batch_processing_queue(
    cache_dir=cache_dir,
    model_id=model_id,
    input_dir=input_dir,
    device=device,
    chunks=True,
    language="en",
    output_dir=output_dir,
)
print(results)
