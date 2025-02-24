"""
https://huggingface.co/docs/transformers/main_classes/pipelines#pipeline-batching
https://huggingface.co/docs/datasets/en/audio_dataset
https://huggingface.co/docs/transformers/en/model_doc/whisper
Batch Processing: It only works:
(1) under gpu with gpu has enough resources
(2) files in the batch has uniform length
(3) the number of columns of inputs should not be large
"""

# Could not do long-form audio transcription: need pass attention_mask
# difference between audio.wav and sample_data.wav
# each batch only contains 30 seconds
# Could do batch processing even for one audio file
# If the file is too long, do batch processing of one file
# If the files are short (< 30 seconds), put them together to do batch
# processing.

from transformers import pipeline, WhisperProcessor
from datasets import Dataset, Audio

dataset = Dataset.from_dict(
    {
        "audio": [
            "/Users/jf3375/Desktop/asr_api/data/audio.wav",
            "/Users/jf3375/Desktop/asr_api/data/sample_data.wav",
        ]
    }
).cast_column("audio", Audio())
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
transcriber = pipeline(
    model="openai/whisper-tiny", device=0, batch_size=2
)  # Divide four files into 2

audio_filenames = [dataset[0]["audio"], dataset[0]["audio"]]
texts = transcriber(audio_filenames)
print(texts)

audio_filenames = [dataset[1]["audio"], dataset[1]["audio"]]
texts = transcriber(audio_filenames)
print(texts)

# Could not do batch processing for two long functions
# The built-in function codes do not pass attention_mask
# audio_filenames = [dataset[0]['audio'], dataset[1]['audio']]
