# Song matching

CLAP + LangChain song retrieval API (see project plan for the full pipeline).

## Run the API

From the repo root (so Python can import the `app` package):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Health: `GET http://localhost:8000/health`
- OpenAPI: `http://localhost:8000/docs`

When you implement CLAP + LangChain indexing, add those packages with `pip` as you go (e.g. `torch`, `transformers`, `langchain-core`, `faiss-cpu`).
