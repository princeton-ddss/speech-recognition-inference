import os
import subprocess
from typing import Optional
from huggingface_hub import snapshot_download, login, list_repo_commits


def download_hf_models(
    models: list[str],
    hf_access_token: Optional[str],
    cache_dir: Optional[str] = None,
) -> None:
    """Download models from the Hugging Face model hub.

    Accepts an optional Hugging Face token for gated model access.

    Args
        - models: a list of model repos, e.g., ["openai/whisper-tiny"]
        - hf_access_token: an optional Hugging Face access token.
        - cache_dir: the directory to store model files to (default: ~/.cache/huggingface/hub)
    """

    if hf_access_token is not None:
        login(token=hf_access_token)

    for repo_id in models:
        snapshot_download(repo_id=repo_id, cache_dir=cache_dir)


def get_latest_commit(repo_id: str, revisions: list[str]) -> str:  # pragma: no cover
    """Return the most recent revision for a model from a list of options."""
    if len(revisions) == 0:
        raise Exception("List of revisions should be non-empty.")
    commits = map(lambda x: x.commit_id, list_repo_commits(repo_id))
    for commit in commits:
        if commit in revisions:
            return revisions[revisions.index(commit)]
    raise Exception("List of revisions should be a (non-empty) subset of repo commits.")


def convert_video_to_audio(file_path, file_name, audio_file_path=None):
    """
    If a audio_file_path is not provided,
    by default the audio would be saved in the same directory as its
    video file with the same file name
    """
    video_file_path = os.path.join(file_path, file_name)
    file_name_noftype = file_name.split(".")[0]
    if not audio_file_path:
        audio_file_path = os.path.join(file_path, file_name_noftype + ".wav")
    command = "ffmpeg -hide_banner -loglevel error -y -i {} -vn {}".format(
        video_file_path, audio_file_path
    )
    subprocess.call(command, shell=True)
