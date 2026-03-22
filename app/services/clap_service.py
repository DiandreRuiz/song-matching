"""
CLAP service: owns model lifecycle and exposes audio/text embedding methods.

Used by build_index.py (audio tower) and the /match endpoint (text tower).
"""

from __future__ import annotations

import numpy as np
import torch
import librosa
from concurrent.futures import ThreadPoolExecutor
from transformers import ClapModel, ClapProcessor
from app.config import get_settings

class ClapService:
    """load CLAP model/processor, implement embed_audio and embed_text."""

    def __init__(self, model_id: str) -> None:
        self.pre_processor = ClapProcessor.from_pretrained(model_id)
        self.model = ClapModel.from_pretrained(model_id)
        self.model.eval()

    def _embed_waveform(self, waveform: np.ndarray, sample_rate: int, chunk_seconds: int) -> np.ndarray:
        """Split a waveform into fixed-length chunks and return the averaged embedding."""
        chunk_size = sample_rate * chunk_seconds
        chunks = [waveform[i:i + chunk_size] for i in range(0, len(waveform), chunk_size)]

        # Run each chunk through the audio tower and collect embeddings
        chunk_embeddings = []
        for chunk in chunks:
            inputs = self.pre_processor(audios=chunk, sampling_rate=sample_rate, return_tensors="pt")
            with torch.no_grad():
                out = self.model.get_audio_features(**inputs)
                chunk_embeddings.append(out.pooler_output.squeeze().numpy())

        return np.mean(chunk_embeddings, axis=0)
    
    def embed_audio(self, audio_paths: list[str], sample_rate: int = 48000, chunk_seconds: int = 10, max_workers: int = 4) -> list[tuple[str, np.ndarray]]:
        """Embed one or more audio files of any length.

        Loads files in parallel via threads (I/O bound), then embeds each
        sequentially (CPU bound). Accepts a single path or a list of paths.
        Returns a list of (path, embedding) tuples preserving input order.
        """
        def _load(path: str) -> tuple[str, np.ndarray]:
            waveform, _ = librosa.load(path, sr=sample_rate)
            return path, waveform

        # Load and resample all audio files concurrently
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            loaded = list(pool.map(_load, audio_paths))

        # Embed each file by chunking its waveform and averaging the chunk embeddings
        results = []
        for path, waveform in loaded:
            embedding = self._embed_waveform(waveform, sample_rate, chunk_seconds)
            results.append((path, embedding))

        return results


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
    results = clap.embed_audio([tom_path])
    for path, embedding in results:
        print(f"{path}: {embedding}")
