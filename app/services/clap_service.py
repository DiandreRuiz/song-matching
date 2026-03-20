"""
CLAP service: owns model lifecycle and exposes audio/text embedding methods.

Used by build_index.py (audio tower) and the /match endpoint (text tower).
"""

from __future__ import annotations


class ClapService:
    """TODO: load CLAP model/processor, implement embed_audio and embed_text."""

    def __init__(self, model_id: str) -> None:
        raise NotImplementedError

    def embed_audio(self, audio_path: str, sample_rate: int = 48000):
        raise NotImplementedError

    def embed_text(self, text: str):
        raise NotImplementedError
