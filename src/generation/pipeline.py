from pathlib import Path
from ..retrieval.retriever import retrieve, STORE_PATH, IDS_PATH
from ..models.base import EmbeddingModel

CONFIDENCE_THRESHOLD = 0.5


def run_query(
    query: str,
    songs: list,
    model: EmbeddingModel,
    k: int = 5,
    threshold: float = CONFIDENCE_THRESHOLD,
    store_path: Path = STORE_PATH,
    ids_path: Path = IDS_PATH,
) -> tuple[list, bool]:
    """Returns (results, low_confidence) where results is a list of (song, sim, explanation) tuples."""
    raw = retrieve(query, songs, model=model, k=k, store_path=store_path, ids_path=ids_path)
    best_score = raw[0][1] if raw else 0.0
    low_confidence = best_score < threshold
    results = [
        (song, sim, f"[confidence: {sim:.2f}] semantic match to your query")
        for song, sim in raw
    ]
    return results, low_confidence
