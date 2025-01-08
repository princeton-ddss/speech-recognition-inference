"""
https://huggingface.co/docs/transformers/main_classes/pipelines#pipeline-batching
https://huggingface.co/docs/datasets/en/audio_dataset
https://huggingface.co/docs/transformers/en/model_doc/whisper
Batch Processing: It only works:
(1) under gpu with gpu has enough resources
(2) files in the batch has uniform length
(3) the number of columns of inputs should not be large

Not API; But Codes
"""

#Could not do long-form audio transcription: need pass attention_mask
#difference between audio.wav and sample_data.wav
#each batch only contains 30 seconds
#Could do batch processing even for one audio file
#If the file is too long, do batch processing of one file
#If the files are short (< 30 seconds), put them together to do batch
# processing.

#data proessing and batch processing

from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
from datasets import Dataset, Audio

dataset = Dataset.from_dict({"audio": [
    "/Users/jf3375/Desktop/asr_api/data/audio.wav",
    "/Users/jf3375/Desktop/asr_api/data/sample_data.wav"]}).cast_column(
    "audio", Audio())
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-tiny")

# deafult is one batch
# raw batch processing
# two batches
audio_0 = dataset[0]['audio']
audio_1 = dataset[1]['audio']

input_features_0 = processor(audio_0['array'], sampling_rate=audio_0['sampling_rate'],
                             return_tensors="pt",
                             return_attention_mask=True)

input_features_1 = processor(audio_1['array'], sampling_rate=audio_1['sampling_rate'],
                             return_tensors="pt",
                             return_attention_mask=True)

input_features = processor([audio_0['array'], audio_1['array']],
                            sampling_rate=audio_1['sampling_rate'],
                             return_tensors="pt",
                             return_attention_mask=True,
                           predict_timestamps=True)

forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="en", task="transcribe"
)

#Generate Transcriptions without timestamps
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
prompt_ids = processor.get_prompt_ids("thread",
                                      return_tensors="pt")
# prompt_ids = processor.get_prompt_ids(prompt, return_tensors="pt").to(device)

predicted_ids = model.generate(input_features=input_features.input_features,
                               attention_mask=input_features.attention_mask,
                               prompt_ids=prompt_ids,
        forced_decoder_ids=forced_decoder_ids)
transcriptions_batch_nots = processor.batch_decode(predicted_ids,
                                        skip_special_tokens=True)
print(transcriptions_batch_nots[0])
print(transcriptions_batch_nots[1])

ninputs = 2
results_batch_nots=[None]*ninputs
results_batch_nots[0]={'text': transcriptions_batch_nots[0]}
results_batch_nots[1]={'text': transcriptions_batch_nots[1]}

#Generate TimeStamps and Segments
#https://huggingface.co/docs/transformers/en/model_doc/whisper

predicted_ids = model.generate(input_features=input_features.input_features,
                               attention_mask=input_features.attention_mask,
                               return_timestamps=True,
                               forced_decoder_ids=forced_decoder_ids)
print(predicted_ids)
transcriptions_batch = processor.batch_decode(predicted_ids,
                                        skip_special_tokens=True,
                                        decode_with_timestamps=True,
                                        decode_with_language=True)
print(transcriptions_batch[0])
print(transcriptions_batch[1])

ninputs = 2
results_batch=[None]*ninputs
results_batch[0]={'text': transcriptions_batch[0]}
results_batch[1]={'text': transcriptions_batch[1]}



#Users testing

#Security; overuse of results
#Fine-tuning
#Models up and fine-tuning



