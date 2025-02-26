import os
import re
import gc
import torch
from datasets import Dataset, Audio
from typing import Optional
from batch_processing.parsing import parse_string_into_result, parse_results_to_csv
from batch_processing.pipeline import load_model
from batch_processing.chunking import chunking_dir, merge_chunks_results
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
        logger.info(result_string)
        results_batch[idx] = parse_string_into_result(result_string, language)
    if output_dir:
        for idx, result in enumerate(results_batch):
            logger.info(audio_paths[idx])
            logger.info(result)
            output_name = os.path.basename(audio_paths[idx]).split(".")[0]
            logger.info(output_name)
            parse_results_to_csv(result, output_dir, output_name)
    logger.info("Finish batch processing")
    return results_batch


def run_batch_processing_queue(
    cache_dir: str,
    model_id: str,
    input_dir: str,
    batch_size: int,
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
        input_dir = chunking_dir(input_dir)
    else:
        logger.warning(
            "Check to make sure all audio files in the directory                      "
            " are less than 30 seconds. The model would only                      "
            " transcribe first 30 seconds for each file"
        )
    # Create a queue for batch processing of chunks of audio files

    def extract_parts(filename):
        match = re.search(r'_(\d+)\.mp3$', filename)
        if match:
            name_part = filename[:match.start()]
            number_part = int(match.group(1))
            return (name_part, number_part)
        return (filename, 0)

    chunk_files = os.listdir(input_dir)
    chunk_files.sort(key=extract_parts)  # Make sure chunks of same files get processed
    # together
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
    return results_batch


# Run on laptop
# input_dir = "/Users/jf3375/Desktop/asr_api/data/single"
# device = "mps"
# output_dir = "/Users/jf3375/Desktop/asr_api/output/localview_test"
# cache_dir = "/Users/jf3375/Princeton Dropbox/Junying Fang/asr_api/models/Whisper_hf"
# model_id = "openai/whisper-tiny"
#
# results = run_batch_processing_queue(
#     cache_dir=cache_dir,
#     model_id=model_id,
#     model_size_gb=10,
#     input_dir=input_dir,
#     batch_size=1,
#     device=device,
#     chunking=False,
#     language="en",
#     output_dir=output_dir,
# )
# print(results)

# Test on Della Login Node
# input_dir="/scratch/gpfs/jf3375/evaluation_data/data/AMI/wav/chunks"#This has
# # 11GB
# # files
# device="cuda:0" #gpu memory is 10GB
# output_dir="/scratch/gpfs/jf3375/asr_api/output"
# cache_dir = "/scratch/gpfs/jf3375/asr_api/models/Whisper_hf"
# model_id = "openai/whisper-tiny" #model has around 5gb
# total_memory_gb=9.5 #In reality, only has 9.5 gb

# Test on MIG Node
# input_dir="/scratch/gpfs/jf3375/evaluation_data/data/AMI/wav/chunks"#This has
# input_dir = "/scratch/gpfs/jf3375/asr_api/data/test/chunks"
# device="cuda:0" #gpu memory is 10GB
# output_dir="/scratch/gpfs/jf3375/asr_api/output/test"
# cache_dir = "/scratch/gpfs/jf3375/asr_api/models/Whisper_hf"
# model_id = "openai/whisper-tiny" #model has around 1gb
# total_memory_gb=9.5 #In reality, only has 9.5 gb
#
# #batch size is calcualted via empical testing
# results = run_batch_processing_queue(
#     cache_dir=cache_dir,
#     model_id=model_id,
#     model_size_gb=0,
#     input_dir=input_dir,
#     batch_size = 10,
#     device=device,
#     chunking=False,
#     language="en",
#     output_dir=output_dir,
#     total_memory_gb=total_memory_gb
# )
# print(results)

#Test on gpu80 node
input_dir="/scratch/gpfs/jf3375/evaluation_data/data/AMI/wav/chunks"#This has
input_dir = "/scratch/gpfs/jf3375/asr_api/data/test/chunks"
device="cuda:0" #gpu memory is 10GB
output_dir="/scratch/gpfs/jf3375/asr_api/output/test"
cache_dir = "/scratch/gpfs/jf3375/asr_api/models/Whisper_hf"
model_id = "openai/whisper-large-v3" #model has around 1gb
total_memory_gb=80 #In reality, only has 9.5 gb

#batch size is calcualted via empical testing
results = run_batch_processing_queue(
    cache_dir=cache_dir,
    model_id=model_id,
    model_size_gb=0,
    input_dir=input_dir,
    batch_size = 30,
    device=device,
    chunking=False,
    language="en",
    output_dir=output_dir,
    total_memory_gb=total_memory_gb
)
print(results)