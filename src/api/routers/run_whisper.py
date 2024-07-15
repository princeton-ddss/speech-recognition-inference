import whisper_timestamped as whisper
import torch
from typing import Union
def whisper_transcription(
            file:str,
            whisper_model,
            language: Union[str, None] = None
):
        '''
        Run OpenAI Whisper to get transcription texts with timestamp
        adjustments
        - **file**: A path to the audio file
        - **model**: A path to the whisper model
        - **language**: Language in the audio; If not provided, Whisper would
        autodetect the language
        '''

        # By default use gpu device
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
            )

        audio = whisper.load_audio(file)

        if not language:
            # detect the spoken language
            mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio)).to(
                device
                )
            _, probs = whisper_model.detect_language(mel)
            language = max(probs, key=probs.get)

        # Translate text into Whisper
        result = whisper.transcribe(
            whisper_model, audio, beam_size=5, best_of=5,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            vad='auditok', language=language
            )
        return result

