"""
CLAP service: owns model lifecycle and exposes audio/text embedding methods.

Used by build_index.py (audio tower) and the /match endpoint (text tower).
"""

from __future__ import annotations

import numpy as np
import torch
import librosa
from transformers import ClapModel, ClapProcessor
from app.config import get_settings

class ClapService:
    """load CLAP model/processor, implement embed_audio and embed_text."""

    def __init__(self, model_id: str) -> None:
        self.pre_processor = ClapProcessor.from_pretrained(model_id)
        self.model = ClapModel.from_pretrained(model_id)
        self.model.eval()

    def embed_audio(self, audio_path: str, sample_rate: int = 48000, chunk_seconds: int = 10) -> np.ndarray:
        """Embed an audio file of any length by splitting into chunks and averaging.

        Works for both short clips (single chunk) and full songs (multiple chunks).
        """
        waveform, _ = librosa.load(audio_path, sr=sample_rate)
        chunk_size = sample_rate * chunk_seconds
        chunks = [waveform[i:i + chunk_size] for i in range(0, len(waveform), chunk_size)]

        chunk_embeddings = []
        for chunk in chunks:
            inputs = self.pre_processor(audio=chunk, sample_rate=sample_rate, return_tensors="pt")
            with torch.no_grad():
                out = self.model.get_audio_features(**inputs)
                chunk_embeddings.append(out.pooler_output.squeeze().numpy())

        return np.mean(chunk_embeddings, axis=0)

    def embed_text(self, text: str) -> np.ndarray:
        inputs = self.pre_processor(text=[text], return_tensors="pt", padding=True)
        with torch.no_grad():
            out = self.model.get_text_features(**inputs)
            embedding = out.pooler_output
            
        return embedding.squeeze().numpy()
    
    
if __name__ == "__main__":
    settings = get_settings()
    clap = ClapService(settings.clap_model_id)
    tom_path = "tom_drum.wav"
    audio_embedding = clap.embed_audio(tom_path)
    print(audio_embedding)
