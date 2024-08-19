import os
import whisper_timestamped as whisper

# Load model and file inputs from docker run from users
# Need to do the volume mount in docker run
# If the environment variables are not set up, running api on local computer
# Use the default one

print(os.getcwd())
default_input_folder = "/Users/jf3375/Princeton Dropbox/Junying Fang/asr_api/data"
default_model_folder = "/Users/jf3375/Princeton Dropbox/Junying Fang/asr_api/models"
# default_input_folder = "/scratch/gpfs/jf3375/asr_api/data"
# default_model_folder = "/scratch/gpfs/jf3375/asr_api/models"



default_model_name = "whisper"
default_model_size = "test"

model_folder = os.getenv("MODEL_FOLDER", default_model_folder)
model_name = os.getenv("MODEL_NAME", default_model_name)
model_size = os.getenv("MODEL_SIZE", default_model_size)
input_folder = os.getenv("INPUT_FOLDER", default_input_folder)

# Load model during API set up
if model_name == "whisper":
    if model_size == "test":
        model_dir = os.path.join(model_folder, "Whisper", "tiny.pt")
        print("Load The Whisper Model of the Tiny Size")
    if model_size == "large":
        model_dir = os.path.join(model_folder, "Whisper", "large-v2.pt")
        print("Load The Whisper Model of the Large Size")
else:
    raise Exception("The input model name is not consistent with the model \
                    this API is deploying: whisper")

whisper_model = whisper.load_model(model_dir)