import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from .recommender import load_songs

MODEL_NAME = "all-MiniLM-L6-v2"
STORE_PATH = Path("data/vector_store.npy")
IDS_PATH   = Path("data/vector_store_ids.json")


def _song_to_text(song: dict) -> str:
    return (
        f"{song['title']} by {song['artist']}. "
        f"Genre: {song['genre']}. Mood: {song['mood']}. "
        f"Energy: {song['energy']:.2f}, tempo: {song['tempo_bpm']} BPM, "
        f"valence: {song['valence']:.2f}, danceability: {song['danceability']:.2f}, "
        f"acousticness: {song['acousticness']:.2f}."
    )


def build_vector_store(
    csv_path: str = "data/songs.csv",
    store_path: Path = STORE_PATH,
    ids_path: Path = IDS_PATH,
) -> None:
    songs = load_songs(csv_path)
    model = SentenceTransformer(MODEL_NAME)

    texts      = [_song_to_text(s) for s in songs]
    embeddings = model.encode(texts, show_progress_bar=True)

    store_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(store_path, embeddings)
    ids_path.write_text(json.dumps([s["id"] for s in songs]))

    print(f"Vector store built: {len(songs)} songs → {store_path}")


def retrieve(
    query: str,
    songs: list,
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

    model         = SentenceTransformer(MODEL_NAME)
    query_vec     = model.encode([query])[0]

    norms         = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec)
    similarities  = np.dot(embeddings, query_vec) / np.where(norms == 0, 1e-8, norms)

    top_indices   = np.argsort(similarities)[::-1][:k]
    songs_by_id   = {s["id"]: s for s in songs}

    return [(songs_by_id[song_ids[i]], float(similarities[i])) for i in top_indices]
