def calculate_batch_size(model_size_gb, max_file_size_mb=15,
                         remaining_proportion=0.6,
                         device=None,
                         total_memory_gb=None):
    """
    Dynamically Calculate Batch Size in Real Time
    remaining_proportion: remaining memory of gpu not used for batch processing
    """
    # Dynamically calculate batch size based on available resources
    if not device:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if device == "cuda:0" or device == "cuda":
        # Get the total memory
        if not total_memory_gb:
            total_memory = torch.cuda.get_device_properties(0).total_memory
        else:
            total_memory = total_memory_gb * 1e9
        print("Total memory in GB:", total_memory//1e9)
        # Get the allocated memory
        allocated_memory = torch.cuda.memory_allocated(0)
        allocated_memory = max(allocated_memory, model_size_gb) #Consider model size
        print("Allocated memory in GB:", allocated_memory//1e9)
        # Get the cached memory
        cached_memory = torch.cuda.memory_reserved(0)
        print("Cached memory in GB:", cached_memory//1e9)
        # Calculate the free memory
        # Multiply the free memory by 80% to have enough buffer
        free_memory = (
            total_memory - (allocated_memory +cached_memory+model_size_gb*1e9)
        ) * (1-remaining_proportion)
        free_memory_mb = free_memory / 1e6
        print("Free memory in GB", free_memory//1e9)
    elif device == "mps":
        # Get the recommend max memory
        total_memory = torch.mps.recommended_max_memory()
        # Get the allocated memory, which includes cached memory
        used_memory = torch.mps.driver_allocated_memory()
        # Calculate the free memory
        # Multiply the free memory by 80% to have enough buffer
        free_memory = (total_memory - used_memory-model_size_gb*1e9) * \
                      (1-remaining_proportion)
        free_memory_mb = free_memory / 1e6

    if device == "cpu":
       raise Exception("Please choose gpu device for batch processing")
    else:
        batch_size = int(free_memory_mb // max_file_size_mb)
    print("estimated batch size", batch_size)
    if batch_size < 1:
        raise Exception("The current gpu memory is not enough to run the \
        current transcription model. Please either increase gpu memory or \
                        downsize the model")
    return batch_size



