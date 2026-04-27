import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.generation.pipeline import run_query, CONFIDENCE_THRESHOLD

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

# One unit-vector per song — cosine similarity equals dot product
FAKE_EMBEDDINGS = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
], dtype=np.float32)


@pytest.fixture
def store(tmp_path):
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
# Return structure
# ---------------------------------------------------------------------------

def test_run_query_returns_tuple(store):
    sp, ip = store
    model = _mock_model(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    results, low_confidence = run_query(
        "late night studying", SONGS, model=model, k=2, store_path=sp, ids_path=ip
    )
    assert isinstance(results, list)
    assert isinstance(low_confidence, bool)


def test_run_query_result_shape(store):
    sp, ip = store
    model = _mock_model(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    results, _ = run_query(
        "chill beats", SONGS, model=model, k=2, store_path=sp, ids_path=ip
    )
    assert len(results) == 2
    song, sim, explanation = results[0]
    assert isinstance(song, dict) and "title" in song
    assert isinstance(sim, float)
    assert "[confidence:" in explanation


# ---------------------------------------------------------------------------
# Confidence threshold
# ---------------------------------------------------------------------------

def test_high_confidence_no_warning(store):
    """Query vector perfectly aligned with index 0 → similarity = 1.0 → no warning."""
    sp, ip = store
    model = _mock_model(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    _, low_confidence = run_query(
        "late night lofi studying", SONGS, model=model, k=3, store_path=sp, ids_path=ip
    )
    assert low_confidence is False


def test_low_confidence_triggers_warning(store):
    """Adversarial query — zero vector → similarity near 0 → low confidence flagged."""
    sp, ip = store
    adversarial_vec = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    model = _mock_model(adversarial_vec)
    _, low_confidence = run_query(
        "trap angry", SONGS, model=model, k=3,
        threshold=CONFIDENCE_THRESHOLD, store_path=sp, ids_path=ip
    )
    assert low_confidence is True


def test_custom_threshold_respected(store):
    """A score of 1.0 is below a threshold of 1.1 → should flag low confidence."""
    sp, ip = store
    model = _mock_model(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    _, low_confidence = run_query(
        "test", SONGS, model=model, k=1, threshold=1.1, store_path=sp, ids_path=ip
    )
    assert low_confidence is True
