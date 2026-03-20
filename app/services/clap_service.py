"""
CLAP service: owns model lifecycle and exposes audio/text embedding methods.

Used by build_index.py (audio tower) and the /match endpoint (text tower).
"""

from __future__ import annotations

import numpy as np
import torch
import librosa
from transformers import ClapModel, ClapProcessor

class ClapService:
    """load CLAP model/processor, implement embed_audio and embed_text."""

    def __init__(self, model_id: str) -> None:
        self.processor = ClapProcessor.from_pretrained(model_id)
        self.model = ClapModel.from_pretrained(model_id)
        self.model.eval()

    def embed_audio(self, audio_path: str, sample_rate: int = 48000) -> np.ndarray:
        waveform, _ = librosa.load(audio_path, sr=sample_rate)
        inputs = self.processor(audios=waveform, sample_rate=sample_rate, return_tensors="pt")
        with torch.no_grad():
            embedding = self.model.get_audio_features(**inputs)
        
        return embedding.squeeze().numpy()

    def embed_text(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        with torch.no_grad():
            embedding = self.model.get_text_features(**inputs)
        return embedding.squeeze().numpy()
