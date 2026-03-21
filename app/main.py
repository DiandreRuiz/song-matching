import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from app.config import get_settings
from app.services.clap_service import ClapService

import faiss
from fastapi import FastAPI
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


class MatchItem(BaseModel):
    path: str
    score: float
    metadata: dict = Field(default_factory=dict)


class MatchResponse(BaseModel):
    feeling: str
    matches: list[MatchItem]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/match", response_model=MatchResponse)
def match_mood(body: MoodRequest) -> MatchResponse:
    """Retrieve top-k tracks for a natural-language mood query (retrieval not wired yet)."""
    _ = get_settings()
    return MatchResponse(feeling=body.feeling, matches=[])
