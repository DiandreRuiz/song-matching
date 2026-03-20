"""
LangChain `Embeddings` adapter for CLAP `get_text_features` / optional text documents.

Audio vectors for the index are computed offline; this class backs `embed_query` at request time.
"""


def build_clap_embeddings() -> None:  # pragma: no cover - scaffolding
    """Construct a `langchain_core.embeddings.Embeddings` CLAP instance (TODO)."""
    raise NotImplementedError("Install optional `ml` extras and implement CLAP loading.")
