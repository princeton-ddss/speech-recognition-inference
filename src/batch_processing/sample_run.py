from main import run_batch_processing_queue

# Test on MIG Node
input_dir = "/scratch/gpfs/jf3375/asr_api/data/test"
input_chunks_dir = "/scratch/gpfs/jf3375/asr_api/data/test/chunks"
device = "cuda:0"  # gpu memory is 10GB
output_dir = "/scratch/gpfs/jf3375/asr_api/output/test"
cache_dir = "/scratch/gpfs/jf3375/asr_api/models/Whisper_hf"
model_id = "openai/whisper-tiny"  # model has around 1gb
total_memory_gb = 9.5  # In reality, only has 9.5 gb for mig
# batch size is calculated via empical testing
batch_size = 60

results = run_batch_processing_queue(
    cache_dir=cache_dir,
    model_id=model_id,
    input_dir=input_dir,
    output_dir=output_dir,
    input_chunks_dir=input_chunks_dir,
    device=device,
    chunking=False,
    language="en",
    batch_size=batch_size,
    total_memory_gb=total_memory_gb,
)
print(results)
