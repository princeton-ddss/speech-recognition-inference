import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from src.data_types.outputs.transcription import Segment, TranscriptionOutputs

default_input_folder = "/Users/jf3375/Princeton Dropbox/Junying Fang/asr_api/data"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# model_id = "openai/whisper-large-v3"
#
model_id = "openai/whisper-tiny"
model_folder= "/Users/jf3375/Princeton Dropbox/Junying " \
          "Fang/asr_api/models/Whisper_hf"
model_folder = os.path.join(model_folder,
                            "models--openai--whisper-tiny/snapshots")
snapshot_id = [i for i in os.listdir(model_folder) if not i.startswith('.')][0]
model_id = os.path.join(model_folder, snapshot_id)

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

file_name = "sample_data.wav"
sample = os.path.join(default_input_folder, file_name)

# result = pipe(sample, return_timestamps=True, return_language=True)
result = pipe(sample, return_timestamps=True, return_language=True,
              generate_kwargs={"language": "english"})

print(result["text"])

output = TranscriptionOutputs(file=file_name, language='eng')

segments = [None] * len(result["chunks"])
for idx, seg in enumerate(result["chunks"]):
    segments[idx] = Segment(
        language = seg["language"], text=seg["text"], start=seg[
            "timestamp"][0], end=seg["timestamp"][1]
    )

output.segments = segments
print(segments)