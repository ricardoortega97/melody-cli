import json
import logging
import pytest
from pathlib import Path


@pytest.fixture(autouse=True)
def reset_logger():
    logger = logging.getLogger("melody_cli")
    logger.handlers.clear()
    yield
    logger.handlers.clear()


_SONG = {
    "id": 1, "title": "Test Song", "artist": "Artist",
    "genre": "lofi", "mood": "chill", "energy": 0.3,
    "tempo_bpm": 80.0, "valence": 0.5, "danceability": 0.4, "acousticness": 0.7,
}


def test_log_query_session_writes_entry(tmp_path, monkeypatch):
    import src.logger as log_mod
    monkeypatch.setattr(log_mod, "LOG_PATH", tmp_path / "test.log")

    results = [(_SONG, 0.82, "[confidence: 0.82] semantic match")]
    log_mod.log_query_session("late night studying", results, low_confidence=False)

    lines = (tmp_path / "test.log").read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["event"] == "query"
    assert entry["query"] == "late night studying"
    assert entry["low_confidence"] is False
    assert entry["warning"] is None
    assert entry["results"][0]["id"] == 1
    assert entry["results"][0]["score"] == 0.82
    assert "ts" in entry


def test_log_query_session_low_confidence(tmp_path, monkeypatch):
    import src.logger as log_mod
    monkeypatch.setattr(log_mod, "LOG_PATH", tmp_path / "test.log")

    results = [(_SONG, 0.3, "[confidence: 0.30] semantic match")]
    log_mod.log_query_session("aggressive trap music", results, low_confidence=True)

    entry = json.loads((tmp_path / "test.log").read_text().strip())
    assert entry["low_confidence"] is True
    assert entry["warning"] == "LOW CONFIDENCE"
    assert entry["best_score"] == 0.3


def test_log_mode_session_writes_entry(tmp_path, monkeypatch):
    import src.logger as log_mod
    monkeypatch.setattr(log_mod, "LOG_PATH", tmp_path / "test.log")

    results = [(_SONG, 1.75, "Recommended because it matches genre.")]
    log_mod.log_mode_session("genre_first", "Sunday Morning", results)

    entry = json.loads((tmp_path / "test.log").read_text().strip())
    assert entry["event"] == "mode"
    assert entry["mode"] == "genre_first"
    assert entry["profile"] == "Sunday Morning"
    assert entry["results"][0]["id"] == 1
    assert entry["results"][0]["score"] == 1.75


def test_log_exception_writes_traceback(tmp_path, monkeypatch):
    import src.logger as log_mod
    monkeypatch.setattr(log_mod, "LOG_PATH", tmp_path / "test.log")

    try:
        raise ValueError("catalog not found")
    except ValueError as e:
        log_mod.log_exception("load_songs", e)

    entry = json.loads((tmp_path / "test.log").read_text().strip())
    assert entry["event"] == "error"
    assert entry["error"] == "ValueError"
    assert "catalog not found" in entry["message"]
    assert "Traceback" in entry["traceback"]
    assert entry["context"] == "load_songs"


def test_multiple_entries_appended(tmp_path, monkeypatch):
    import src.logger as log_mod
    monkeypatch.setattr(log_mod, "LOG_PATH", tmp_path / "test.log")

    results = [(_SONG, 0.75, "match")]
    log_mod.log_query_session("query one", results, low_confidence=False)
    log_mod.log_mode_session("mood_first", "Late-Night Studier", results)

    lines = (tmp_path / "test.log").read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["event"] == "query"
    assert json.loads(lines[1])["event"] == "mode"
