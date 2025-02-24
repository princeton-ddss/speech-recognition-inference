import os
import re
from pydub import AudioSegment, utils
import shutil
import pandas as pd


def chunking_file(audio_path, input_chunks_dir, chunk_len_in_secs):
    """
    Chunk the audio file into chunks of a fixed lengths
    """
    # Load Audio File
    audio = AudioSegment.from_file(audio_path)

    # Extract audio_dir and audio_file_name
    _, file_name = os.path.dirname(audio_path), os.path.basename(audio_path)

    # No need to chunk if audio file has length <= chunk_len_in_secs
    if len(audio) <= chunk_len_in_secs * 1e3:
        shutil.copy(audio_path, os.path.join(input_chunks_dir, file_name))
    else:
        # Extract file extension
        file_name_no_ext, ext = os.path.splitext(file_name)

        # Split audio into chunks of chunk length
        chunks = utils.make_chunks(audio, chunk_len_in_secs * 1e3)

        # Export each chunk
        for i, chunk in enumerate(chunks):
            chunk.export(
                os.path.join(input_chunks_dir, file_name_no_ext + "_" + str(i) + ext)
            )


def chunking_dir(input_dir, chunk_len_in_secs=30):
    """
    input_dir: input directory of all files
    chunk_len_in_secs: audio length of each chunk in seconds
    """
    input_files = os.listdir(input_dir)
    input_files = [f for f in input_files if not f.startswith(".")]

    # Create directory to save chunk files
    input_chunks_dir = os.path.join(input_dir, "chunks")
    if not os.path.exists(input_chunks_dir):
        os.makedirs(input_chunks_dir)
    else:
        raise Exception(
            f"Directory {input_chunks_dir} already exists. Please remove it or choose a"
            " different name."
        )

    # Chunk input files
    for input_file in input_files:
        chunking_file(
            os.path.join(input_dir, input_file),
            input_chunks_dir,
            chunk_len_in_secs=chunk_len_in_secs,
        )
    return input_chunks_dir


def merge_chunks_results(output_dir):
    chunk_names = [c for c in os.listdir(output_dir) if c.endswith(".csv")]
    chunk_names.sort()  # Sort chunks by the timestamps order
    # Get the Mapping from Parent File to Children File
    pattern = re.compile(r"_\d+\.csv$")
    parent_children_mapping = {}
    for chunk_name in chunk_names:
        parent_name = pattern.sub(".csv", chunk_name)
        if parent_name not in parent_children_mapping:
            parent_children_mapping[parent_name] = [chunk_name]
        else:
            parent_children_mapping[parent_name].append(chunk_name)

    for parent in parent_children_mapping.keys():
        df = pd.DataFrame()
        starting_time = 0
        for child in parent_children_mapping[parent]:
            child_df = pd.read_csv(os.path.join(output_dir, child))
            child_df["start"] = child_df["start"] + starting_time
            child_df["end"] = child_df["end"] + starting_time
            df = pd.concat([df, child_df])
            # Delete child file
            os.remove(os.path.join(output_dir, child))
            starting_time += 30  # 30 seconds chunks
        df.to_csv(os.path.join(output_dir, parent), index=False)
    return None
