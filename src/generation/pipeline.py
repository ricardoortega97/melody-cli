from ..retrieval.retriever import retrieve
from ..models.base import EmbeddingModel


def run_query(query: str, songs: list, model: EmbeddingModel, k: int = 5) -> list:
    """Returns top-k (song, similarity_score, explanation) tuples for a natural language query."""
    results = retrieve(query, songs, model=model, k=k)
    return [
        (song, sim, f"similarity {sim:.3f} — semantic match to your query")
        for song, sim in results
    ]
