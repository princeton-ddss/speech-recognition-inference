import os
import torch
import pandas as pd
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
from datasets import Dataset, Audio
from batch_processing.parsing import parse_string_into_result
from batch_processing.logger import logger
from batch_processing import model, processor
from huggingface_hub import snapshot_download, login, list_repo_commits
from pydub import AudioSegment, utils
from collections import deque
import shutil

def chunking_file(audio_path, input_chunks_dir, chunk_len_in_secs):
    """
    Chunk the audio file into chunks of a fixed lengths
    """
    # Load Audio File
    audio = AudioSegment.from_file(audio_path)

    # Extract audio_dir and audio_file_name
    _, file_name = os.path.dirname(audio_path), os.path.basename(
        audio_path)

    # No need to chunk if audio file has length <= chunk_len_in_secs
    if len(audio)<=chunk_len_in_secs*1e3:
        shutil.copy(audio_path, os.path.join(input_chunks_dir, file_name))
    else:
        # Extract file extension
        file_name_no_ext, ext = os.path.splitext(file_name)

        # Split audio into chunks of chunk length
        chunks = utils.make_chunks(audio, chunk_len_in_secs*1e3)

        # Export each chunk
        for i, chunk in enumerate(chunks):
            chunk.export(os.path.join(input_chunks_dir, file_name_no_ext+'_'+str(i)+ext))

def chunking_dir(input_dir, chunk_len_in_secs=1200):
    """
    input_dir: input directory of all files
    chunk_len_in_secs: audio length of each chunk in seconds
    """
    input_files = os.listdir(input_dir)
    input_files = [f for f in input_files if not f.startswith('.')]

    #Create directory to save chunk files
    input_chunks_dir = os.path.join(input_dir, 'chunks')
    if not os.path.exists(input_chunks_dir):
        os.makedirs(input_chunks_dir)
    else:
        raise Exception(
            f"Directory {input_chunks_dir} already exists. Please remove it or choose a different name."
        )

    #Chunk input files
    for input_file in input_files:
        chunking_file(os.path.join(input_dir, input_file), input_chunks_dir,
                                chunk_len_in_secs=chunk_len_in_secs)
    return input_chunks_dir


def batch_processing(
                    model,
                    audio_paths,
                    device=None,
                    language=None,
                     sampling_rate=16000,
     output_dir=None):
    """
    audio_paths: the list of path to audio chunks files for batch processing
    output_dir: save transcription results in csv file in output path
    """

    dataset = Dataset.from_dict({"audio": audio_paths}).\
        cast_column("audio", Audio())
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))
    batch_size = len(dataset)
    batch = [None]*batch_size
    for idx, data in enumerate(dataset):
        batch[idx] = data['audio']['array']

    #Set language if exists
    if language:
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=language, task="transcribe")
    else:
        forced_decoder_ids = None

    #Prepare input features
    input_features = processor(
        batch,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        return_attention_mask=True,
        predict_timestamps=True,
        device=device
        ).to(device)

    predicted_ids = model.generate(
        input_features=input_features.input_features,
        attention_mask=input_features.attention_mask,
        return_timestamps=True,
        forced_decoder_ids=forced_decoder_ids
        )

    if not language:
        language = processor.decode(predicted_ids[0,1], device=device)[2:-2]

    transcriptions_batch = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True,
        decode_with_timestamps=True,
        device=device
        )

    results_batch = [None]*batch_size
    for idx, result_string in enumerate(transcriptions_batch):
        results_batch[idx] = parse_string_into_result(result_string, language)
    if output_dir:
        for idx, result in enumerate(results_batch):
            output_name = os.path.basename(audio_paths[idx]).split('.')[0]
            parse_results_to_csv(result, output_dir, output_name)
    return results_batch

def parse_results_to_csv(result, output_dir, output_name):
    # Translate text into Whisper
    segment_len = len(result['chunks'])
    transcribe_df = pd.DataFrame()
    start_list = [0] * segment_len
    end_list = [0] * segment_len
    text_list = [0] * segment_len

    for idx, segment in enumerate(result['chunks']):
        start_list[idx], end_list[idx], text_list[idx] = segment[
            'timestamp'][0], segment['timestamp'][1], segment['text']

    transcribe_df['start'], transcribe_df['end'], transcribe_df['text'] = start_list, end_list, text_list
    transcribe_df['file_name'] = output_name
    transcribe_df['speaker'] = ''

    # Remove leading and trailing whitespaces from whisper outputs
    transcribe_df['text'] = transcribe_df['text'].apply(lambda x: x.strip())

    transcribe_df.to_csv(os.path.join(output_dir, '{}.csv'.format(
        output_name)), index=False)
    return transcribe_df

def calculate_batch_size(max_file_size_mb=25,
                     buffer_proportion=0.5):
    """
    Dynamically Calculate Batch Size in Real Time
    """
    # Dynamically calculate batch size based on available resources
    device = "mps"
    torch.device(device)
    if device == "cuda":
        # Get the total memory
        total_memory = torch.cuda.get_device_properties(0).total_memory
        # Get the allocated memory
        allocated_memory = torch.cuda.memory_allocated(0)
        # Get the cached memory
        cached_memory = torch.cuda.memory_reserved(0)
        # Calculate the free memory
        #Multiply the free memory by 80% to have enough buffer
        free_memory = (total_memory - (allocated_memory +
                                       cached_memory))*buffer_proportion
        free_memory_mb = free_memory/1e6
    elif device == "mps":
        # Get the recommend max memory
        total_memory = torch.mps.recommended_max_memory()
        # Get the allocated memory, which includes cached memory
        used_memory=torch.mps.driver_allocated_memory()
        # Calculate the free memory
        #Multiply the free memory by 80% to have enough buffer
        free_memory = (total_memory - used_memory)*buffer_proportion
        free_memory_mb = free_memory/1e6

    batch_size = int(free_memory_mb//max_file_size_mb)
    return batch_size

def run_batch_processing_queue(input_dir,
                     chunks = True,
                     language=None,
                     sampling_rate=16000,
                     device=None,
                     output_dir=None):
    """
    This function assumes that the length of each audio file is less than or
    equal to 20 minutes for Whisper to run properly under batch processing.
    output_dir: save transcription results in csv file in output path

    chunks: If True, chunk files into input_dir and save chunks in
    input_chunks_dir.
    If False, input_dir already contains all chunks
    """
    if chunks:
        #Chunk files into input_dir and save chunks in input_chunks_dir
        input_dir = chunking_dir(input_dir)
    # Create a queue for batch processing of chunks of audio files
    chunk_files = os.listdir(input_dir)
    chunk_files_path = [os.path.join(input_dir, chunk_file) for
                        chunk_file in chunk_files if not chunk_file.startswith('.')]
    chunks_queue = deque(chunk_files_path)

    #Load Model on Device
    if not device:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Process audio chunk files under batch processing
    audio_paths = []
    while chunks_queue:
        batch_size = calculate_batch_size(max_file_size_mb=25,
                     buffer_proportion=0.5)
        for _ in range(batch_size):
            if chunks_queue:
                audio_paths.append(chunks_queue.pop())
            else:
                break
        results_batch = batch_processing(model, audio_paths, device, language,
                               sampling_rate, output_dir)
    return results_batch



input_dir="/Users/jf3375/Desktop/asr_api/data/localview_test/chunks"
device="mps"
output_dir="/Users/jf3375/Desktop/asr_api/output/localview_test"
results =  run_batch_processing_queue(input_dir,
                                      chunks=False,
                                      language="en",
                                      device=device,
                                      output_dir=output_dir)
print(results)