import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from app.config import get_settings
from app.services.clap_service import ClapService

import numpy as np
import faiss
from fastapi import Depends, FastAPI, Request
from pydantic import BaseModel, Field





@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    
    # Load CLAP model
    _app.state.clap = ClapService(settings.clap_model_id)
    
    # Load pre-built FAISS index and path mapping
    store = Path(settings.vector_store_path)
    _app.state.index = faiss.read_index(str(store / "index.faiss"))
    with open(store / "paths.json") as f:
        _app.state.paths = json.load(f)
    yield


app = FastAPI(
    title="Song Matching",
    description="Mood text → CLAP text tower → vector search over precomputed audio embeddings",
    lifespan=lifespan,
)


class MoodRequest(BaseModel):
    feeling: str = Field(..., min_length=1, max_length=4000)
    k: int = Field(default=3, ge=1, le=20)


class MatchItem(BaseModel):
    path: str
    score: float
    metadata: dict = Field(default_factory=dict)


class MatchResponse(BaseModel):
    feeling: str
    matches: list[MatchItem]


def get_clap(request: Request) -> ClapService:
    return request.app.state.clap

def get_index(request: Request) -> faiss.IndexFlatIP:
    return request.app.state.index

def get_paths(request: Request) -> list[str]:
    return request.app.state.paths


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/match", response_model=MatchResponse)
def match_mood(
    body: MoodRequest,
    clap: ClapService = Depends(get_clap),
    index: faiss.IndexFlatIP = Depends(get_index),
    paths: list[str] = Depends(get_paths),
) -> MatchResponse:
    """Retrieve top-k tracks for a natural-language mood query."""
    # Embed mood text into a query vector and reshape for FAISS (expects 2D)
    query_embedding = clap.embed_text(body.feeling).reshape(1, -1)

    # Search the index for the k nearest audio embeddings by similarity
    similarity_scores, faiss_indices = index.search(query_embedding, body.k)

    # Map FAISS positions back to song paths; skip -1 (empty slots when k > index size)
    matches = [
        MatchItem(path=paths[faiss_idx], score=float(similarity_score))
        for similarity_score, faiss_idx in zip(similarity_scores[0], faiss_indices[0])
        if faiss_idx != -1
    ]

    return MatchResponse(feeling=body.feeling, matches=matches)
