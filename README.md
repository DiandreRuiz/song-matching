# Song Matching

Match natural-language mood descriptions to songs using [CLAP](https://huggingface.co/docs/transformers/model_doc/clap) audio/text embeddings and [FAISS](https://github.com/facebookresearch/faiss) vector search.

Describe a vibe in plain English and the API returns the closest-matching songs from your library, ranked by cosine similarity.

## How It Works

1. **Offline indexing** — `build_index.py` loads every song in a directory, splits each into 10-second chunks, embeds them through CLAP's audio tower, averages the chunk embeddings per song, L2-normalizes, and stores everything in a FAISS inner-product index.
2. **Live matching** — The `/match` endpoint embeds your mood text through CLAP's text tower (same shared vector space), queries the FAISS index, and returns the top-k songs by similarity score.

## Prerequisites

- Python 3.11+
- A directory of audio files (`.mp3`, `.wav`, or `.flac`)

## Setup

```bash
git clone <repo-url> && cd song-matching
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
VECTOR_STORE_PATH=./data/vector_store
CLAP_MODEL_ID=laion/clap-htsat-unfused
SONGS_DIR_PATH=./songs
```

| Variable | Description |
|---|---|
| `VECTOR_STORE_PATH` | Where the FAISS index and path mapping are saved |
| `CLAP_MODEL_ID` | HuggingFace model ID for the CLAP checkpoint |
| `SONGS_DIR_PATH` | Directory containing your audio files |

Place your audio files in the `songs/` directory (or wherever `SONGS_DIR_PATH` points).

## Build the Index

This scans your audio directory, computes embeddings, and saves the FAISS index:

```bash
python -m scripts.build_index
```

You can also point to a different audio directory:

```bash
python -m scripts.build_index --audio-dir /path/to/music
```

Re-run this whenever you add or remove songs, or change the CLAP model.

## Start the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Health check: `GET http://localhost:8000/health`
- Interactive docs: `http://localhost:8000/docs`

## Query the API

**POST** `/match`

```json
{
  "feeling": "dark and moody with heavy bass",
  "k": 3
}
```

- `feeling` — A natural-language description of the mood or sonic character you're looking for.
- `k` — Number of matches to return (1–20, default 3).

**Response:**

```json
{
  "feeling": "dark and moody with heavy bass",
  "matches": [
    { "path": "songs/basement_door.mp3", "score": 0.5563, "metadata": {} },
    { "path": "songs/black_coffee_at_midnight.mp3", "score": 0.4203, "metadata": {} },
    { "path": "songs/get_down.mp3", "score": 0.3994, "metadata": {} }
  ]
}
```

Scores are cosine similarities (range -1 to 1). Higher is a closer match.

## Run the Test Suite

With the server running in another terminal:

```bash
python -m scripts.test_match
```

This sends several mood descriptions (aggressive rap, bright acoustic, heavy metal, horror, love ballad) and prints the top matches for each.

## Project Structure

```
song-matching/
├── app/
│   ├── config.py              # Pydantic settings (loads .env)
│   ├── main.py                # FastAPI app, lifespan, /match endpoint
│   └── services/
│       └── clap_service.py    # CLAP model wrapper (embed_audio, embed_text)
├── scripts/
│   ├── build_index.py         # Offline: build FAISS index from audio files
│   └── test_match.py          # Smoke tests for the /match endpoint
├── data/
│   └── vector_store/          # Generated index.faiss + paths.json
├── songs/                     # Your audio files (git-ignored)
├── requirements.txt
└── .env                       # Config overrides (git-ignored)
```

## Tips

- **CLAP doesn't understand lyrics.** It processes raw audio waveforms — instruments, tempo, energy, vocal timbre. Write mood descriptions in terms of what the music *sounds like*, not what it's about.
- **Short prompts work best.** "dark and moody" scores higher than "Music that sounds dark and moody" with the default model.
- **Rebuild the index after model changes.** The audio and text embeddings must come from the same CLAP checkpoint to be comparable.
