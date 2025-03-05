import torch
from typing import Optional
def calculate_batch_size(max_file_matrix_size_mb=20,
                         remaining_proportion=0.6,
                         device=None,
                         total_memory_gb: Optional[int] = None
):
    """
    Dynamically Calculate Batch Size in Real Time
    remaining_proportion: remaining memory of gpu not used for batch processing
    """
    if not device:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
       raise Exception("Please choose gpu device for batch processing")

    # Get the total memory
    if not total_memory_gb:
        total_memory = torch.cuda.get_device_properties(0).total_memory \
            if "cuda" in device else torch.mps.recommended_max_memory()
    else:
        total_memory = total_memory_gb * 1e9
    print("Total memory in GB:", total_memory // 1e9)

    # Get the allocated memory
    allocated_memory = torch.cuda.memory_allocated(0) if "cuda" in device \
        else torch.mps.driver_allocated_memory()
    print("Allocated memory in GB:", allocated_memory // 1e9)

    # Get the cached memory
    cached_memory = torch.cuda.memory_reserved(0) if "cuda" in device else 0
    print("Cached memory in GB:", cached_memory // 1e9)

    # Calculate the free memory
    free_memory = (
        total_memory - (allocated_memory+cached_memory)
    ) * (1-remaining_proportion)
    free_memory_mb = free_memory / 1e6

    #Calculate the batch size
    print("Free Memory in MB:", free_memory_mb)

    batch_size = int(free_memory_mb // max_file_matrix_size_mb)
    print("estimated batch size", batch_size)
    if batch_size < 1:
        raise Exception("The current gpu memory is not enough to run the \
        current transcription model. Please either increase gpu memory or \
                        downsize the model")
    return batch_size



