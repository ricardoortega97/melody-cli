## MelodyCLI

**Project name:** ai110-music-rec-sim (now melodyCLI)

ai110-music-rec-sim was a CLI music recommender that scored each song in a catalog against a user's stated taste profile — genre, mood, and audio features (energy, tempo, valence, acousticness) — using a point-weight system.

Songs were filtered for eligibility, scored across three signals (genre match +2.0, mood match +1.0, vibe closeness up to +1.5), then ranked and returned as a top-K list. A session feedback loop tracked skips and dislikes, applying cooldowns and disqualifications that narrowed future recommendations over time.

---

## Title and Summary

**melodyCLI — RAG-Powered Music Recommendation CLI**
melodyCLI upgrades the original rule-based recommender with a RAG layer, letting users describe intent in plain language instead of selecting fixed genre or mood labels. The system embeds each query, runs semantic vector search over the song catalog, and sends the top-*k* matches to the existing scorer, which then applies audio-feature ranking to produce the final recommendations.

The result is a CLI that accepts inputs like `"late night studying"` or `"high energy gym warmup"` and returns ranked, explained recommendations — with a confidence warning when the query falls outside the catalog's coverage.

---

## System Architecture

### Data Flow

### Where Humans Are Involved

- **Setup:** User provides natural language query via `--query` flag.
- **Evaluation:** Human reviews sample interactions to verify semantic relevance.
- **Catalog review:** Human audits songs flagged as low-confidence to determine if catalog expansion is needed.

---

## Architecture Overview

_Short explanation of the system diagram — to be written._

---

## Setup Instructions

**Requirements:** Python 3.10+

```bash
# 1. Clone the repo
git clone <repo-url> && cd melody-cli

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API key
cp .env.example .env   # then fill in your key

# 5. Build the vector store (run once)
python -m src.main --build

# 6. Run
python -m src.main --query "late night studying"
python -m src.main --mode genre_first
```

---

## Commands

| Flag | Description |
|------|-------------|
| `--build` | Embed `songs.csv` into the vector store. Run once before using `--query`. |
| `--query "<text>"` | Natural language search — returns top 5 songs ranked by semantic similarity. |
| `--mode <strategy>` | Rule-based mode using hardcoded profiles. Choices: `genre_first` (default), `mood_first`, `energy_focused`. |

```bash
python -m src.main --build
python -m src.main --query "late night studying"
python -m src.main --mode mood_first
```

`--query` and `--mode` are independent paths. `--query` uses the RAG pipeline; `--mode` uses the original rule-based scorer.

---

## Sample Interactions

_At least 2–3 examples of inputs and resulting AI outputs — to be filled in after Milestone 2._

---

## Design Decisions

_Why you built it this way and what trade-offs you made — to be written._