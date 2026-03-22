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

def _l2_normalize(v: np.ndarray) -> np.ndarray:
    """Scale a vector to unit length so inner product equals cosine similarity."""
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm


class ClapService:
    """load CLAP model/processor, implement embed_audio and embed_text."""

    def __init__(self, model_id: str) -> None:
        """Load a CLAP model and processor from HuggingFace and set to inference mode."""
        self.pre_processor = ClapProcessor.from_pretrained(model_id)
        self.model = ClapModel.from_pretrained(model_id)
        self.model.eval()

    def embed_audio(self, audio_paths: list[str], sampling_rate: int = 48000, chunk_seconds: int = 10) -> list[tuple[str, np.ndarray]]:
        """Embed one or more audio files of any length.

        Loads files sequentially, then embeds all chunks in a single batched
        forward pass. Returns a list of (path, embedding) tuples preserving
        input order.
        """
        # Load and resample all audio files
        loaded = []
        for path in audio_paths:
            waveform, _ = librosa.load(path, sr=sampling_rate)
            loaded.append((path, waveform))

        # Embed each file's chunks in its own batch to avoid cross-file padding skew
        results = []
        for path, waveform in loaded:
            chunk_size = sampling_rate * chunk_seconds
            min_chunk_size = chunk_size // 2
            chunks = [waveform[i:i + chunk_size] for i in range(0, len(waveform), chunk_size)]
            chunks = [c for c in chunks if len(c) >= min_chunk_size]

            inputs = self.pre_processor(audios=chunks, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
            with torch.no_grad():
                out = self.model.get_audio_features(**inputs)
                projected = self.model.audio_projection(out.pooler_output)
                chunk_embeddings = projected.detach().numpy()

            file_embedding = _l2_normalize(np.mean(chunk_embeddings, axis=0))
            results.append((path, file_embedding))

        return results


    def embed_text(self, text: str) -> np.ndarray:
        """Embed a text string via CLAP's text tower into a normalized projected vector."""
        inputs = self.pre_processor(text=[text], return_tensors="pt", padding=True)
        with torch.no_grad():
            out = self.model.get_text_features(**inputs)
            projected = self.model.text_projection(out.pooler_output)

        return _l2_normalize(projected.squeeze().detach().numpy())
    
    
if __name__ == "__main__":
    settings = get_settings()
    clap = ClapService(settings.clap_model_id)
    tom_path = "tom_drum.wav"
    results = clap.embed_audio([tom_path])
    for path, embedding in results:
        print(f"{path}: {embedding}")
