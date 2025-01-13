import os
import torch
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
from datasets import Dataset, Audio
from batch_processing.parsing import parse_string_into_result
from batch_processing.logger import logger
from batch_processing import model, processor
from huggingface_hub import snapshot_download, login, list_repo_commits


def batch_processing(audio_paths, model, processor,
                     language=None,
                     revision=None,
                     sampling_rate=16000,
                     device=None):
    """
    This function assumes that the length of each audio file is less than or
    equal to 20 minutes for Whisper to run properly under batch processing.
    """
    #Run on specific device such as gpu
    if not device:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Prepare Batch
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
    return results_batch

audio_paths = ["/Users/jf3375/Desktop/asr_api/data/audio.wav",
    "/Users/jf3375/Desktop/asr_api/data/sample_data.wav"]

results = batch_processing(audio_paths, model, processor, language="fr")
print(results)