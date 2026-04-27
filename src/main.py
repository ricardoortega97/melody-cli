"""
Command line runner for the Music Recommender Simulation.
"""

import argparse
import textwrap

from dotenv import load_dotenv
load_dotenv()

from .generation.recommender import load_songs, recommend_songs, SCORING_MODES
from .generation.pipeline import run_query


USER_PROFILES = [
    # Late-night study session: low energy lofi, chilled out
    {"name": "Late-Night Studier", "genre": "lofi",     "mood": "chill",    "energy": 0.3},
    # High-intensity workout: hard rock, intense mood, max energy
    {"name": "Gym Grinder",        "genre": "rock",     "mood": "intense",  "energy": 0.9},
    # Sunday morning wind-down: jazz, relaxed vibe, moderate energy
    {"name": "Sunday Morning",     "genre": "jazz",     "mood": "relaxed",  "energy": 0.4},
    # ADVERSARIAL: genre/mood not in dataset — zero categorical hits, ranking falls to vibe only
    {"name": "Ghost Listener",     "genre": "trap",     "mood": "angry",    "energy": 0.5},
    # ADVERSARIAL: contradicts itself — lofi/chill genre+mood but max energy; categorical bonus (+3.0) drowns out vibe signal (+1.5 max)
    {"name": "Mismatch Maximizer", "genre": "kpop",     "mood": "chill",    "energy": 0.97},
]


def _print_recommendations(user_prfs: dict, recommendations: list) -> None:
    # Column content widths (excludes 1-space padding on each side)
    W = {"rank": 3, "title": 20, "artist": 14, "style": 20, "score": 5, "why": 38}
    COLS = list(W.values())

    H, V = "─", "│"
    total_width = sum(w + 2 for w in COLS) + len(COLS) + 1

    def _border(left, mid, right):
        return left + mid.join(H * (w + 2) for w in COLS) + right

    def _trunc(s: str, n: int) -> str:
        s = str(s)
        return s if len(s) <= n else s[:n - 1] + "…"

    def _data_row(rank, title, artist, style, score, why_lines):
        """Returns one or more table row strings for a single entry (wraps the Why column)."""
        cells = [
            f" {rank :<{W['rank']}} ",
            f" {title:<{W['title']}} ",
            f" {artist:<{W['artist']}} ",
            f" {style:<{W['style']}} ",
            f" {score:>{W['score']}} ",
        ]
        prefix = V + V.join(cells) + V
        blank  = V + V.join(" " * (w + 2) for w in COLS[:-1]) + V

        first = f"{prefix} {(why_lines[0] if why_lines else ''):<{W['why']}} {V}"
        rest  = [f"{blank} {line:<{W['why']}} {V}" for line in why_lines[1:]]
        return [first] + rest

    name_label = user_prfs.get("name", "User")
    if user_prfs.get("_query_mode"):
        profile_label = f"query: \"{user_prfs.get('_query', '')}\""
    else:
        profile_label = f"{user_prfs['genre']}  ·  {user_prfs['mood']}  ·  energy {user_prfs['energy']}"

    print()
    print("═" * total_width)
    print(f" {name_label}  ·  Top {len(recommendations)}  ·  {profile_label}")
    print("═" * total_width)
    print(_border("┌", "┬", "┐"))
    for line in _data_row("#", "Title", "Artist", "Genre · Mood", "Score", ["Why"]):
        print(line)
    print(_border("├", "┼", "┤"))

    for i, (song, score, explanation) in enumerate(recommendations, start=1):
        style  = _trunc(f"{song['genre']} · {song['mood']}", W["style"])
        reason = explanation
        if reason.startswith("Recommended because it "):
            reason = reason[len("Recommended because it "):]
        reason = reason.rstrip(".")
        why_lines = textwrap.wrap(reason, width=W["why"]) or ["—"]

        for line in _data_row(
            f"#{i}",
            _trunc(song["title"],  W["title"]),
            _trunc(song["artist"], W["artist"]),
            style,
            f"{score:.2f}",
            why_lines,
        ):
            print(line)

        if i < len(recommendations):
            print(_border("├", "┼", "┤"))

    print(_border("└", "┴", "┘"))


def _build_model(choice: str):
    """Instantiate the chosen EmbeddingModel. Uses lazy imports to avoid loading heavy deps on --mode path."""
    if choice == "gemini":
        from .models.gemini import GeminiEmbeddingModel
        return GeminiEmbeddingModel()
    from .models.local import LocalEmbeddingModel
    return LocalEmbeddingModel()


def main() -> None:
    parser = argparse.ArgumentParser(description="Music Recommender Simulation")
    parser.add_argument(
        "--mode",
        choices=list(SCORING_MODES),
        default="genre_first",
        help="Scoring strategy to rank songs by. Choices: " + ", ".join(SCORING_MODES),
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Embed songs.csv into the vector store and exit.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Natural language query to retrieve and rank songs via RAG.",
    )
    parser.add_argument(
        "--model",
        choices=["local", "gemini"],
        default="local",
        help="Embedding backend to use with --query or --build (default: local).",
    )
    args = parser.parse_args()

    if args.build:
        from .ingestion.builder import build_vector_store
        model = _build_model(args.model)
        build_vector_store(model)
        return

    songs = load_songs("data/songs.csv")

    if args.query:
        model = _build_model(args.model)
        recommendations, low_confidence = run_query(args.query, songs, model=model, k=5)
        if low_confidence:
            print("\n⚠  LOW CONFIDENCE — best match below threshold; consider refining your query or expanding the catalog.")
        user_prfs = {"name": "RAG Results", "_query_mode": True, "_query": args.query}
        _print_recommendations(user_prfs, recommendations)
        return

    print(f"\nScoring mode: {args.mode}\n")

    for user_prfs in USER_PROFILES:
        recommendations = recommend_songs(user_prfs, songs, k=5, mode=args.mode)
        _print_recommendations(user_prfs, recommendations)


if __name__ == "__main__":
    main()
