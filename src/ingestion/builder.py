import json
from pathlib import Path

import numpy as np

from ..generation.recommender import load_songs
from ..models.base import EmbeddingModel
from ..retrieval.retriever import STORE_PATH, IDS_PATH, store_paths, _song_to_text


def build_vector_store(
    model: EmbeddingModel,
    csv_path: str = "data/songs.csv",
    store_path: Path = STORE_PATH,
    ids_path: Path = IDS_PATH,
    model_key: str | None = None,
) -> None:
    if model_key:
        store_path, ids_path = store_paths(model_key)
    songs      = load_songs(csv_path)
    texts      = [_song_to_text(s) for s in songs]
    embeddings = model.encode(texts)

    store_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(store_path, embeddings)
    ids_path.write_text(json.dumps([s["id"] for s in songs]))

    print(f"Vector store built: {len(songs)} songs → {store_path}")
