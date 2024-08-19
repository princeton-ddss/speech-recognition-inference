import os

from huggingface_hub import snapshot_download
from huggingface_hub import login

# Log into huggingface
def download_hf_model(download_model_path, hf_access_token, model_folder =
    'Whisper_hf', model_repo_ids =['openai/whisper-tiny']):
    login(token=hf_access_token)

    for model_repo_id in model_repo_ids:
        # Download LLaMA Model to Local Folder
        snapshot_download(repo_id = model_repo_id,  cache_dir= os.path.join(download_model_path, model_folder))

# Download model locally
# download_model_path = "/Users/jf3375/Princeton Dropbox/Junying Fang/asr_api/models"
# model_folder = "Whisper_hf"
# hf_access_token = "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"
#
# download_hf_model(download_model_path, hf_access_token)

# Download model on Della
download_model_path = "/scratch/gpfs/jf3375/asr_api/models"
model_folder = "Whisper_hf"
hf_access_token = "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"

download_hf_model(download_model_path, hf_access_token, model_repo_ids =
["openai/whisper-tiny", "openai/whisper-large-v3"])