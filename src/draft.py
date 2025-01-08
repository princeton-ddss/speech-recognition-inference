import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, Pipeline, pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

revision_dir = "/Users/jf3375/Princeton Dropbox/Junying Fang/asr_api/models/Whisper_hf/models--openai--whisper-tiny/snapshots/169d4a4341b33bc18d8881c4b69c2e104e1cc0af"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    revision_dir,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
)

model.to(device)

processor = AutoProcessor.from_pretrained(revision_dir)

transcriber = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device
    )

transcription = transcriber("/Users/jf3375/Desktop/asr_api/data/audio.wav")
transcription_timestamp = transcriber(
    "/Users/jf3375/Desktop/asr_api/data/audio.wav",
                            return_timestamps=True)

print(transcription)
print(transcription_timestamp)