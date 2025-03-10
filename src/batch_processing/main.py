import os
import re
import gc
import torch
from datasets import Dataset, Audio
from typing import Optional
from batch_processing.parsing import parse_string_into_result, parse_results_to_csv
from batch_processing.pipeline import load_model
from batch_processing.chunking import chunking_dir, merge_chunks_results
from batch_processing.batch_size import calculate_batch_size
from collections import deque
from logger import logger

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
    logger.info("Try to Run Batch Processing")
    logger.info(audio_paths)
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
    logger.info("Finish batch processing")
    return results_batch


def run_batch_processing_queue(
    cache_dir: str,
    model_id: str,
    input_dir: str,
    input_chunks_dir: Optional[str]=None,
    batch_size: Optional[int]=None,
    total_memory_gb: Optional[int]=None,
    max_file_matrix_size_mb: Optional[int] = 100,
    remaining_memory_proportion: Optional[int] = 0.6,
    revision: Optional[str] = None,
    hf_access_token: Optional[str] = None,
    device: Optional[str]=None,
    chunking=True,
    language=None,
    sampling_rate=16000,
    output_dir=None,
    rerun = False
):
    """
    This function assumes that the length of each audio file is less than or
    equal to 20 minutes for Whisper to run properly under batch processing.
    output_dir: save transcription results in csv file in output path

    chunking: If True, chunk files into input_dir and save chunks in
    input_chunks_dir.
    If False, input_dir already contains all chunks

    rerun: If True, the model would be rerun on all files in input
    directory. If False, the model would only be run on files in input
    directory which do not have outputs yet.
    """
    # Consider conditions where input_chunks_dir does not exist
    if not input_chunks_dir:
        if chunking:
            # Chunk files into input_dir and save chunks in input_chunks_dir
            input_chunks_dir = chunking_dir(input_dir)
        else:
            raise Exception("Please either set chunking as true or pass a"
                            "directory with chunk files")
    else:
            logger.warning(
                "Check to make sure all audio files in the chunking "
                "directory"
                " are less than 30 seconds. The model would only"
                " transcribe first 30 seconds for each file"
            )
    chunk_files = os.listdir(input_chunks_dir)

    # If rerun is false, only run models on audio files which do not have
    # existing outputs
    if not rerun:
        chunk_files_results = [output.split('.')[0] for output in os.listdir(output_dir)]
        if set(chunk_files_results)==set([input.split('.')[0] for input in os.listdir(input_dir) if input != "chunks"]):
            logger.info("All the input files already get processed")
            return
        else:
            chunk_files = [file for file in chunk_files if file.split(
                '.')[0] not in chunk_files_results]

    # Load Model
    model, processor = load_model(
        cache_dir=cache_dir,
        model_id=model_id,
        revision=revision,
        hf_access_token=hf_access_token,
    )


    # Create a queue for batch processing of chunks of audio files

    def extract_parts(filename):
        match = re.search(r'_(\d+)\.mp3$', filename)
        if match:
            name_part = filename[:match.start()]
            number_part = int(match.group(1))
            return (name_part, number_part)
        return (filename, 0)

    chunk_files.sort(key=extract_parts)  # Make sure chunks of same files get processed
    # together
    chunk_files_path = [
        os.path.join(input_chunks_dir, chunk_file)
        for chunk_file in chunk_files
        if not chunk_file.startswith(".")
    ]
    chunks_queue = deque(chunk_files_path)

    # Load Model on Device
    if not device:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Ensure that the model is put into the device before calculating batch
    # size
    torch.cuda.synchronize(device)

    # Calculate the batch size if it is None
    if not batch_size:
        batch_size = calculate_batch_size(
            max_file_matrix_size_mb=max_file_matrix_size_mb,
            remaining_proportion=remaining_memory_proportion,
            total_memory_gb=total_memory_gb,
            device=device
        )

    # Process audio chunk files under batch processing
    nruns = 1
    audio_paths = []
    while chunks_queue:
        for _ in range(batch_size):
            if chunks_queue:
                audio_paths.append(chunks_queue.popleft())
            else:
                break
        results_batch = batch_processing(
            model, processor, audio_paths, device, language, sampling_rate, output_dir
        )
        logger.info("nruns:{}".format(nruns))
        logger.info("batch size:{}".format(batch_size))
        nruns+=1
        #Empty inputs and cache for the next iteration of batch processing
        audio_paths = []
        torch.cuda.empty_cache()
        gc.collect()

    # Merge results of chunks of one file together
    merge_chunks_results(output_dir)
    logger.info("Batch Processing Done")
    return


