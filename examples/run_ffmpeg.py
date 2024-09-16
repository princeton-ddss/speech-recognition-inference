import os
import subprocess


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


default_input_folder = "/Users/jf3375/'Princeton Dropbox'/'Junying Fang'/asr_api/data"
file_name = "sample_data.mov"

convert_video_to_audio(default_input_folder, file_name)
