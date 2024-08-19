import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Load model and file inputs from docker run from users
# Need to do the volume mount in docker run
# If the environment variables are not set up, running api on local computer
# Use the default one

print(os.getcwd())
# default_input_folder = "/Users/jf3375/Princeton Dropbox/Junying Fang/asr_api/data"
# default_model_folder = "/Users/jf3375/Princeton Dropbox/Junying Fang/asr_api/models/Whisper_hf"
default_input_folder = "/scratch/gpfs/jf3375/asr_api/data"
default_model_folder = "/scratch/gpfs/jf3375/asr_api/models/Whisper_hf"



default_model_name = "whisper"
default_model_size = "large"

model_folder = os.getenv("MODEL_FOLDER", default_model_folder)
model_name = os.getenv("MODEL_NAME", default_model_name)
model_size = os.getenv("MODEL_SIZE", default_model_size)
input_folder = os.getenv("INPUT_FOLDER", default_input_folder)

print("model_folder", model_folder)

# Load model during API set up
if model_name == "whisper":
    if model_size == "test":
        model_folder = os.path.join(model_folder,
                                    "models--openai--whisper-tiny/snapshots")
        snapshot_id = [i for i in os.listdir(model_folder) if not i.startswith('.')][0]
        model_id = os.path.join(model_folder, snapshot_id)
        print("Start Loading The Whisper Model of the Tiny Size")
    if model_size == "large":
        model_folder = os.path.join(model_folder,
                                    "models--openai--whisper-large-v3/snapshots")
        snapshot_id = [i for i in os.listdir(model_folder) if not i.startswith('.')][0]
        model_id = os.path.join(model_folder, snapshot_id)
        print("Start Loading The Whisper Model of the Large Size")
else:
    raise Exception("The input model name is not consistent with the model \
                    this API is deploying: whisper")

device = "cuda:0" if torch.cuda.is_available() else "cpu"

torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

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

print("Finished Loading Models")