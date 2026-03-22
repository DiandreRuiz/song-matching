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
        """Load a CLAP model and processor from HuggingFace and set to inference mode."""
        self.pre_processor = ClapProcessor.from_pretrained(model_id)
        self.model = ClapModel.from_pretrained(model_id)
        self.model.eval()

    def embed_audio(self, audio_paths: list[str], sample_rate: int = 48000, chunk_seconds: int = 10, max_workers: int = 4) -> list[tuple[str, np.ndarray]]:
        """Embed one or more audio files of any length.

        Loads files in parallel via threads (I/O bound), then embeds all
        chunks in a single batched forward pass (CPU/GPU bound).
        Returns a list of (path, embedding) tuples preserving input order.
        """
        def _load(path: str) -> tuple[str, np.ndarray]:
            """Read an audio file from disk and resample to the target rate."""
            waveform, _ = librosa.load(path, sr=sample_rate)
            return path, waveform

        # Load and resample all audio files concurrently
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            loaded = list(pool.map(_load, audio_paths))

        # Split each waveform into chunks, tracking which file each chunk belongs to
        all_chunks = []
        chunk_to_file = []
        for file_idx, (_, waveform) in enumerate(loaded):
            chunk_size = sample_rate * chunk_seconds
            chunks = [waveform[i:i + chunk_size] for i in range(0, len(waveform), chunk_size)]
            all_chunks.extend(chunks)
            chunk_to_file.extend([file_idx] * len(chunks))

        # Batch all chunks into a single forward pass through the audio tower
        inputs = self.pre_processor(audios=all_chunks, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = self.model.get_audio_features(**inputs)
            all_embeddings = out.pooler_output.numpy()

        # Average the chunk embeddings per file
        results = []
        for file_idx, (path, _) in enumerate(loaded):
            mask = [i for i, f in enumerate(chunk_to_file) if f == file_idx]
            file_embedding = np.mean(all_embeddings[mask], axis=0)
            results.append((path, file_embedding))

        return results


    def embed_text(self, text: str) -> np.ndarray:
        """Embed a text string via CLAP's text tower into a 512-d vector."""
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
