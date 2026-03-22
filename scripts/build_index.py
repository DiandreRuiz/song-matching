"""
Offline: scan audio directory, compute CLAP audio features, persist LangChain vector store.

Usage: python scripts/build_index.py --audio-dir ...
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from app.config import get_settings
from app.services.clap_service import ClapService
import numpy as np
import faiss

def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Build FAISS index from audio files"
    )
    parser.add_argument(
        "--audio-dir",
        required=False,
        help="Path to directory of audio files",
        default=settings.songs_dir_path
    )
    
    args = parser.parse_args()
    audio_dir = Path(args.audio_dir)
    
    # Collect all supported audio files
    audio_files = []
    for ext in ("*.wav", "*.mp3", "*.flac"):
        audio_files.extend(audio_dir.glob(ext))
    
    # Embed all audio files
    clap = ClapService(settings.clap_model_id)
    embeddings = []
    paths = []
    for audio_file in audio_files:
        paths.append(str(audio_file))
        embeddings.append(clap.embed_audio(str(audio_file)))
        
    # Build FAISS index
    embedding_matrix = np.stack(embeddings)
    dimensionality = embedding_matrix.shape[1] # 512
    index = faiss.IndexFlatIP(dimensionality) # Flat (brute force)
    index.add(embedding_matrix)
    
    # Store index & paths to disk
    store_path = Path(settings.vector_store_path)
    store_path.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(store_path / "index.faiss"))
    with open(store_path / "paths.json", "w") as f:
        json.dump(paths, f)
    
    
if __name__ == "__main__":
    main()
