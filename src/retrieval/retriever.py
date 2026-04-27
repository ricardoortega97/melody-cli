import json
from pathlib import Path

import numpy as np

from ..models.base import EmbeddingModel

VECTORS_DIR = Path("vectors")
STORE_PATH  = VECTORS_DIR / "vector_store.npy"
IDS_PATH    = VECTORS_DIR / "vector_store_ids.json"


def store_paths(model_key: str) -> tuple[Path, Path]:
    return (
        VECTORS_DIR / f"vector_store_{model_key}.npy",
        VECTORS_DIR / f"vector_store_{model_key}_ids.json",
    )


def _song_to_text(song: dict) -> str:
    return (
        f"{song['title']} by {song['artist']}. "
        f"Genre: {song['genre']}. Mood: {song['mood']}. "
        f"Energy: {song['energy']:.2f}, tempo: {song['tempo_bpm']} BPM, "
        f"valence: {song['valence']:.2f}, danceability: {song['danceability']:.2f}, "
        f"acousticness: {song['acousticness']:.2f}."
    )


def retrieve(
    query: str,
    songs: list,
    model: EmbeddingModel,
    k: int = 5,
    store_path: Path = STORE_PATH,
    ids_path: Path = IDS_PATH,
) -> list:
    """Returns top-k (song, similarity_score) tuples for the given natural-language query."""
    if not store_path.exists() or not ids_path.exists():
        raise FileNotFoundError(
            f"Vector store not found at {store_path}. Run with --build first."
        )

    embeddings = np.load(store_path)
    song_ids   = json.loads(ids_path.read_text())

    query_vec    = model.encode([query])[0]
    norms        = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec)
    similarities = np.dot(embeddings, query_vec) / np.where(norms == 0, 1e-8, norms)

    top_indices  = np.argsort(similarities)[::-1][:k]
    songs_by_id  = {s["id"]: s for s in songs}

    return [(songs_by_id[song_ids[i]], float(similarities[i])) for i in top_indices]
