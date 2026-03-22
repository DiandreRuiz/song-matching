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
    
    TOTAL_STEPS = 5
    args = parser.parse_args()
    audio_dir = Path(args.audio_dir)

    print(f"\n🕐 [Step 1/{TOTAL_STEPS}] Scanning {audio_dir} for audio files...")
    audio_files = []
    for ext in ("*.wav", "*.mp3", "*.flac"):
        audio_files.extend(audio_dir.glob(ext))
    print(f"✅ [Step 1/{TOTAL_STEPS}] Found {len(audio_files)} audio files\n")

    print(f"🕐 [Step 2/{TOTAL_STEPS}] Loading CLAP model ({settings.clap_model_id})...")
    clap = ClapService(settings.clap_model_id)
    print(f"✅ [Step 2/{TOTAL_STEPS}] CLAP model loaded\n")

    print(f"🕐 [Step 3/{TOTAL_STEPS}] Loading audio & generating embeddings for {len(audio_files)} files...")
    results = clap.embed_audio([str(f) for f in audio_files])
    paths = [path for path, _ in results]
    embeddings = [embedding for _, embedding in results]
    print(f"✅ [Step 3/{TOTAL_STEPS}] Embeddings generated for {len(embeddings)} files\n")

    print(f"🕐 [Step 4/{TOTAL_STEPS}] Building FAISS index...")
    embedding_matrix = np.stack(embeddings)
    dimensionality = embedding_matrix.shape[1]
    index = faiss.IndexFlatIP(dimensionality)
    index.add(embedding_matrix)
    print(f"✅ [Step 4/{TOTAL_STEPS}] FAISS index built ({len(embeddings)} vectors, {dimensionality}d)\n")

    print(f"🕐 [Step 5/{TOTAL_STEPS}] Saving index & paths to {settings.vector_store_path}...")
    store_path = Path(settings.vector_store_path)
    store_path.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(store_path / "index.faiss"))
    with open(store_path / "paths.json", "w") as f:
        json.dump(paths, f)
    print(f"✅ [Step 5/{TOTAL_STEPS}] Saved index.faiss and paths.json\n")

    print(f"🎉 Done! Indexed {len(embeddings)} songs to {settings.vector_store_path}")
    
    
if __name__ == "__main__":
    main()
