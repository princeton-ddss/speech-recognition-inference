from batch_processing.pipeline import load_model

cache_dir="/Users/jf3375/Princeton Dropbox/Junying Fang/asr_api/models/Whisper_hf"
model_id = "openai/whisper-tiny"
model, processor = load_model(cache_dir=cache_dir, model_id=model_id)
