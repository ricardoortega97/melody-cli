from src.generation.recommender import (
    Song, UserProfile, Recommender,
    recommend_songs, score_song, init_session_state,
)
from src.generation.pipeline import CONFIDENCE_THRESHOLD

CATALOG = [
    Song(id=1, title="Test Pop Track",  artist="A", genre="pop",  mood="happy",   energy=0.8, tempo_bpm=120, valence=0.9, danceability=0.8, acousticness=0.2),
    Song(id=2, title="Chill Lofi Loop", artist="B", genre="lofi", mood="chill",   energy=0.4, tempo_bpm=80,  valence=0.6, danceability=0.5, acousticness=0.9),
    Song(id=3, title="Jazz Sunday",     artist="C", genre="jazz", mood="relaxed", energy=0.3, tempo_bpm=90,  valence=0.7, danceability=0.5, acousticness=0.8),
    Song(id=4, title="Rock Workout",    artist="D", genre="rock", mood="intense", energy=0.9, tempo_bpm=140, valence=0.5, danceability=0.7, acousticness=0.1),
]

def make_small_recommender() -> Recommender:
    return Recommender(CATALOG[:2])


# ---------------------------------------------------------------------------
# Original tests
# ---------------------------------------------------------------------------

def test_recommend_returns_songs_sorted_by_score():
    user = UserProfile(favorite_genre="pop", favorite_mood="happy", target_energy=0.8, likes_acoustic=False)
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)
    assert len(results) == 2
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    user = UserProfile(favorite_genre="pop", favorite_mood="happy", target_energy=0.8, likes_acoustic=False)
    rec = make_small_recommender()
    explanation = rec.explain_recommendation(user, rec.songs[0])
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


def test_explanation_is_causal():
    user = UserProfile(favorite_genre="pop", favorite_mood="happy", target_energy=0.8, likes_acoustic=False)
    rec = make_small_recommender()
    explanation = rec.explain_recommendation(user, rec.songs[0])
    assert any(word in explanation.lower() for word in ["because", "matches", "close", "fits"])


# ---------------------------------------------------------------------------
# Confidence-path coverage — verify scoring behavior that feeds pipeline confidence
# ---------------------------------------------------------------------------

def _songs_as_dicts():
    from dataclasses import asdict
    return [asdict(s) for s in CATALOG]


def test_matched_profile_top_score_above_confidence_threshold():
    """A well-matched profile (lofi/chill) should produce a top rule-based score that
    is comfortably above zero, confirming the scoring path that backs high confidence."""
    songs = _songs_as_dicts()
    session = init_session_state(songs)
    tempo_min = min(s["tempo_bpm"] for s in songs)
    tempo_max = max(s["tempo_bpm"] for s in songs)
    lofi_song = next(s for s in songs if s["genre"] == "lofi")
    score = score_song({"genre": "lofi", "mood": "chill", "energy": 0.4}, lofi_song, session, tempo_min, tempo_max)
    assert score > CONFIDENCE_THRESHOLD


def test_adversarial_profile_receives_no_categorical_bonus():
    """Ghost Listener (trap/angry) gets no genre or mood bonus — score is vibe-only.
    Mirrors why the pipeline flags these queries as low confidence."""
    songs = _songs_as_dicts()
    session = init_session_state(songs)
    tempo_min = min(s["tempo_bpm"] for s in songs)
    tempo_max = max(s["tempo_bpm"] for s in songs)
    ghost_prfs = {"genre": "trap", "mood": "angry", "energy": 0.5}

    for song in songs:
        score = score_song(ghost_prfs, song, session, tempo_min, tempo_max)
        assert score <= 1.5, f"{song['title']} scored {score} — expected no categorical bonus"


def test_adversarial_profile_scores_lower_than_matched_profile():
    """Ghost Listener's best score must be lower than Late-Night Studier's best score."""
    songs = _songs_as_dicts()
    session = init_session_state(songs)
    tempo_min = min(s["tempo_bpm"] for s in songs)
    tempo_max = max(s["tempo_bpm"] for s in songs)

    ghost_best = max(
        score_song({"genre": "trap", "mood": "angry", "energy": 0.5}, s, session, tempo_min, tempo_max)
        for s in songs
    )
    studier_best = max(
        score_song({"genre": "lofi", "mood": "chill", "energy": 0.3}, s, session, tempo_min, tempo_max)
        for s in songs
    )
    assert ghost_best < studier_best


def test_all_five_profiles_return_k_results():
    """Every profile — including adversarial ones — must return k results."""
    songs = _songs_as_dicts()
    profiles = [
        {"genre": "lofi",  "mood": "chill",   "energy": 0.3},
        {"genre": "rock",  "mood": "intense",  "energy": 0.9},
        {"genre": "jazz",  "mood": "relaxed",  "energy": 0.4},
        {"genre": "trap",  "mood": "angry",    "energy": 0.5},
        {"genre": "kpop",  "mood": "chill",    "energy": 0.97},
    ]
    for prfs in profiles:
        results = recommend_songs(prfs, songs, k=4)
        assert len(results) == 4, f"Expected 4 results for {prfs}, got {len(results)}"


def test_recommend_songs_result_structure():
    """Each result from recommend_songs must be a (song_dict, float, str) triple —
    the same structure the pipeline produces so both paths render identically."""
    songs = _songs_as_dicts()
    results = recommend_songs({"genre": "pop", "mood": "happy", "energy": 0.8}, songs, k=2)
    for song, score, explanation in results:
        assert isinstance(song, dict) and "title" in song
        assert isinstance(score, float)
        assert isinstance(explanation, str) and explanation.strip() != ""
