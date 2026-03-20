from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.config import get_settings

@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    # Warm settings (validates env); vector store + CLAP load will go here.
    get_settings()
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
