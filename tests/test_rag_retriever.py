import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.retrieval.retriever import _song_to_text, retrieve
from src.ingestion.builder import build_vector_store

SONGS = [
    {
        "id": 1, "title": "Midnight Coding", "artist": "LoRoom",
        "genre": "lofi", "mood": "chill",
        "energy": 0.42, "tempo_bpm": 78, "valence": 0.56,
        "danceability": 0.62, "acousticness": 0.71,
    },
    {
        "id": 2, "title": "Gym Hero", "artist": "Max Pulse",
        "genre": "pop", "mood": "intense",
        "energy": 0.93, "tempo_bpm": 132, "valence": 0.77,
        "danceability": 0.88, "acousticness": 0.05,
    },
    {
        "id": 3, "title": "Coffee Shop Stories", "artist": "Slow Stereo",
        "genre": "jazz", "mood": "relaxed",
        "energy": 0.37, "tempo_bpm": 90, "valence": 0.71,
        "danceability": 0.54, "acousticness": 0.89,
    },
]

# One unit-vector per song so cosine similarity = dot product
FAKE_EMBEDDINGS = np.array([
    [1.0, 0.0, 0.0],   # Midnight Coding
    [0.0, 1.0, 0.0],   # Gym Hero
    [0.0, 0.0, 1.0],   # Coffee Shop Stories
], dtype=np.float32)


@pytest.fixture
def store(tmp_path):
    """Writes fake embeddings to tmp_path and returns (store_path, ids_path)."""
    sp = tmp_path / "store.npy"
    ip = tmp_path / "ids.json"
    np.save(sp, FAKE_EMBEDDINGS)
    ip.write_text(json.dumps([s["id"] for s in SONGS]))
    return sp, ip


def _mock_model(query_vec: np.ndarray) -> MagicMock:
    m = MagicMock()
    m.encode.return_value = np.array([query_vec])
    return m


# ---------------------------------------------------------------------------
# _song_to_text
# ---------------------------------------------------------------------------

def test_song_to_text_contains_metadata():
    text = _song_to_text(SONGS[0])
    assert "Midnight Coding" in text
    assert "lofi" in text
    assert "chill" in text
    assert "LoRoom" in text


# ---------------------------------------------------------------------------
# build_vector_store
# ---------------------------------------------------------------------------

def test_build_creates_store_and_ids(tmp_path):
    sp = tmp_path / "store.npy"
    ip = tmp_path / "ids.json"
    fake_embed = np.random.rand(3, 8).astype(np.float32)

    mock_model = MagicMock()
    mock_model.encode.return_value = fake_embed

    with patch("src.ingestion.builder.load_songs", return_value=SONGS):
        build_vector_store(mock_model, "data/songs.csv", store_path=sp, ids_path=ip)

    assert sp.exists()
    assert ip.exists()
    assert np.load(sp).shape == (3, 8)
    assert json.loads(ip.read_text()) == [1, 2, 3]


# ---------------------------------------------------------------------------
# retrieve
# ---------------------------------------------------------------------------

def test_retrieve_returns_k_results(store):
    sp, ip = store
    query_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    results = retrieve("lofi study", SONGS, model=_mock_model(query_vec), k=2, store_path=sp, ids_path=ip)
    assert len(results) == 2


def test_retrieve_result_is_song_score_tuple(store):
    sp, ip = store
    query_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    results = retrieve("chill beats", SONGS, model=_mock_model(query_vec), k=1, store_path=sp, ids_path=ip)

    song, score = results[0]
    assert isinstance(song, dict) and "title" in song
    assert isinstance(score, float)


def test_retrieve_best_match_for_lofi_query(store):
    sp, ip = store
    # Query vector closest to index 0 (Midnight Coding / lofi chill)
    query_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    results = retrieve("late night studying lofi", SONGS, model=_mock_model(query_vec), k=1, store_path=sp, ids_path=ip)
    assert results[0][0]["id"] == 1


def test_retrieve_best_match_for_workout_query(store):
    sp, ip = store
    # Query vector closest to index 1 (Gym Hero / intense)
    query_vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    results = retrieve("high energy workout pump", SONGS, model=_mock_model(query_vec), k=1, store_path=sp, ids_path=ip)
    assert results[0][0]["id"] == 2


def test_retrieve_raises_when_store_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="--build"):
        retrieve(
            "test query",
            SONGS,
            model=MagicMock(),
            store_path=tmp_path / "missing.npy",
            ids_path=tmp_path / "missing.json",
        )
