from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def _user_to_prfs(self, user: UserProfile) -> Dict:
        """Converts a UserProfile dataclass to the dict format expected by scoring functions."""
        return {
            "genre":  user.favorite_genre,
            "mood":   user.favorite_mood,
            "energy": user.target_energy,
        }

    def _song_to_dict(self, song: Song) -> Dict:
        """Converts a Song dataclass to the dict format expected by scoring functions."""
        from dataclasses import asdict
        return asdict(song)

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Returns the top k songs ranked by score against the given user profile."""
        user_prfs  = self._user_to_prfs(user)
        songs_dict = [self._song_to_dict(s) for s in self.songs]
        session    = init_session_state(songs_dict)

        tempo_min = min(s["tempo_bpm"] for s in songs_dict)
        tempo_max = max(s["tempo_bpm"] for s in songs_dict)

        ranked = sorted(
            [(s, score_song(user_prfs, d, session, tempo_min, tempo_max))
             for s, d in zip(self.songs, songs_dict)],
            key=lambda x: x[1],
            reverse=True,
        )
        return [song for song, _ in ranked[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Returns a human-readable string explaining why a song matched the user profile."""
        user_prfs = self._user_to_prfs(user)
        song_dict = self._song_to_dict(song)
        songs_dict = [self._song_to_dict(s) for s in self.songs]

        tempo_min = min(s["tempo_bpm"] for s in songs_dict)
        tempo_max = max(s["tempo_bpm"] for s in songs_dict)

        return _build_explanation(user_prfs, song_dict, tempo_min, tempo_max)

def load_songs(csv_path: str) -> List[Dict]:
    """Reads songs.csv and returns a list of song dicts with correctly typed numeric fields."""
    import csv
    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            songs.append({
                "id":           int(row["id"]),
                "title":        row["title"],
                "artist":       row["artist"],
                "genre":        row["genre"],
                "mood":         row["mood"],
                "energy":       float(row["energy"]),
                "tempo_bpm":    float(row["tempo_bpm"]),
                "valence":      float(row["valence"]),
                "danceability": float(row["danceability"]),
                "acousticness": float(row["acousticness"]),
            })
    print(f"Loaded {len(songs)} songs:")
    for song in songs:
        print(f"  [{song['id']}] {song['title']} — {song['artist']} ({song['genre']}, {song['mood']})")
    return songs


# ---------------------------------------------------------------------------
# Scoring Modes — weight config dicts for each ranking strategy.
# Each mode controls the categorical bonuses and the per-feature vibe weights.
# ---------------------------------------------------------------------------
SCORING_MODES: Dict[str, Dict] = {
    "genre_first": {
        "genre_bonus":    2.0,
        "mood_bonus":     0.5,
        "energy_w":       1,
        "tempo_w":        1,
        "valence_w":      1,
        "acousticness_w": 1,
    },
    "mood_first": {
        "genre_bonus":    0.5,
        "mood_bonus":     2.0,
        "energy_w":       1,
        "tempo_w":        1,
        "valence_w":      2,
        "acousticness_w": 1,
    },
    "energy_focused": {
        "genre_bonus":    0.5,
        "mood_bonus":     0.5,
        "energy_w":       4,
        "tempo_w":        1,
        "valence_w":      1,
        "acousticness_w": 1,
    },
}


def vibe_closeness(user_prfs: Dict, song: Dict, tempo_min: float, tempo_max: float, weights: Optional[Dict] = None) -> float:
    """Returns 0.0–1.5 score based on weighted mean distance across energy, tempo, valence, and acousticness.
    Per-feature weights come from the active scoring mode (see SCORING_MODES)."""
    wts = weights or SCORING_MODES["genre_first"]
    tempo_range = tempo_max - tempo_min
    song_tempo_norm = (song["tempo_bpm"] - tempo_min) / tempo_range if tempo_range > 0 else 0.5

    pairs: List[Tuple[float, int]] = []

    if "energy" in user_prfs:
        pairs.append((abs(song["energy"] - float(user_prfs["energy"])), wts["energy_w"]))

    if "tempo_bpm" in user_prfs:
        user_tempo_norm = (float(user_prfs["tempo_bpm"]) - tempo_min) / tempo_range if tempo_range > 0 else 0.5
        pairs.append((abs(song_tempo_norm - user_tempo_norm), wts["tempo_w"]))

    if "valence" in user_prfs:
        pairs.append((abs(song["valence"] - float(user_prfs["valence"])), wts["valence_w"]))

    if "acousticness" in user_prfs:
        pairs.append((abs(song["acousticness"] - float(user_prfs["acousticness"])), wts["acousticness_w"]))

    if not pairs:
        return 0.0

    total_weight = sum(w for _, w in pairs)
    mean_distance = sum(d * w for d, w in pairs) / total_weight
    return round(1.5 * (1 - mean_distance), 4)


def score_song(user_prfs: Dict, song: Dict, session_state: Dict, tempo_min: float, tempo_max: float, weights: Optional[Dict] = None) -> float:
    """Scores one song using genre bonus, mood bonus, vibe closeness (+0–1.5), and skip penalty (-0.5).
    Bonus magnitudes and vibe weights are controlled by the active scoring mode (see SCORING_MODES)."""
    wts = weights or SCORING_MODES["genre_first"]
    score = 0.0

    if song["genre"] == user_prfs.get("genre", ""):
        score += wts["genre_bonus"]

    if song["mood"] == user_prfs.get("mood", ""):
        score += wts["mood_bonus"]

    score += vibe_closeness(user_prfs, song, tempo_min, tempo_max, weights=wts)

    song_state = session_state.get(song["id"], {})
    if song_state.get("skip_count", 0) == 1:
        score -= 0.5

    return round(score, 4)


def init_session_state(songs: List[Dict]) -> Dict:
    """Returns a fresh session state with zeroed skip counts, cooldowns, and queue position for all songs."""
    state: Dict = {"queue_position": 0}
    for song in songs:
        state[song["id"]] = {
            "skip_count":    0,
            "disqualified":  False,
            "cooldown_until": 0,
        }
    return state


def is_eligible(song: Dict, session_state: Dict) -> bool:
    """Returns True if a song is neither disqualified nor within its cooldown window."""
    song_state = session_state.get(song["id"], {})
    if song_state.get("disqualified", False):
        return False
    if session_state.get("queue_position", 0) < song_state.get("cooldown_until", 0):
        return False
    return True


def apply_feedback(song_id: int, signal: str, session_state: Dict) -> None:
    """Updates session state for a played song: play = no-op, skip = strike/cooldown, off = immediate disqualify."""
    song_state = session_state.get(song_id, {})

    if signal == "off":
        song_state["disqualified"] = True
    elif signal == "skip":
        song_state["skip_count"] = song_state.get("skip_count", 0) + 1
        if song_state["skip_count"] == 1:
            song_state["cooldown_until"] = session_state.get("queue_position", 0) + 25
        elif song_state["skip_count"] >= 2:
            song_state["disqualified"] = True

    session_state[song_id] = song_state
    session_state["queue_position"] = session_state.get("queue_position", 0) + 1


def recommend_songs(user_prfs: Dict, songs: List[Dict], k: int = 5, session_state: Optional[Dict] = None, mode: str = "genre_first") -> List[Tuple[Dict, float, str]]:
    """Filters eligible songs, scores each one, and returns the top k as (song, score, explanation) tuples.
    Pass mode="mood_first" or mode="energy_focused" to switch ranking strategies (see SCORING_MODES)."""
    if session_state is None:
        session_state = init_session_state(songs)

    weights = SCORING_MODES.get(mode, SCORING_MODES["genre_first"])
    tempo_min = min(s["tempo_bpm"] for s in songs)
    tempo_max = max(s["tempo_bpm"] for s in songs)

    scored = sorted(
        [
            (song, score_song(user_prfs, song, session_state, tempo_min, tempo_max, weights=weights))
            for song in songs
            if is_eligible(song, session_state)
        ],
        key=lambda x: x[1],
        reverse=True,
    )

    return [
        (song, score, _build_explanation(user_prfs, song, tempo_min, tempo_max, weights=weights))
        for song, score in scored[:k]
    ]


def _build_explanation(user_prfs: Dict, song: Dict, tempo_min: float, tempo_max: float, weights: Optional[Dict] = None) -> str:
    """Builds a causal explanation of why a song was recommended, reflecting each scoring component."""
    wts = weights or SCORING_MODES["genre_first"]
    parts = []

    if song["genre"] == user_prfs.get("genre", ""):
        parts.append(f"matches your {song['genre']} genre preference (+{wts['genre_bonus']:.1f})")

    if song["mood"] == user_prfs.get("mood", ""):
        parts.append(f"fits your {song['mood']} mood (+{wts['mood_bonus']:.1f})")

    if "energy" in user_prfs:
        diff = abs(song["energy"] - float(user_prfs["energy"]))
        if diff < 0.15:
            parts.append(
                f"energy is close to yours ({song['energy']:.2f} vs {float(user_prfs['energy']):.2f}, weighted {wts['energy_w']}x in vibe)"
            )

    vibe = vibe_closeness(user_prfs, song, tempo_min, tempo_max, weights=wts)
    parts.append(f"overall vibe score {vibe:.2f}/1.50")

    if not parts:
        return "No strong match — included by vibe proximity only."
    return "Recommended because it " + ", ".join(parts) + "."
