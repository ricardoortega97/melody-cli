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

# 4. (Optional) Add Gemini API key for Gemini embeddings
cp .env.example .env   # set GEMINI_API_KEY — skip if using local model

# 5. Build the vector store (run once; local model requires no API key)
python -m src.main --build
# or with Gemini:
python -m src.main --build --model gemini

# 6. Run
python -m src.main --query "late night studying"
python -m src.main --mode genre_first
```

---

## Commands

| Flag | Values | Description |
|------|--------|-------------|
| `--build` | — | Embed `songs.csv` into the vector store. Run once before using `--query`. |
| `--query "<text>"` | any string | Natural language search — returns top 5 songs ranked by semantic similarity. |
| `--model` | `local` (default), `gemini` | Embedding backend to use with `--query` or `--build`. |
| `--mode <strategy>` | `genre_first` (default), `mood_first`, `energy_focused` | Rule-based mode using hardcoded profiles. |

```bash
# Build the vector store (local model — no API key required)
python -m src.main --build

# Build with Gemini embeddings (requires GEMINI_API_KEY)
python -m src.main --build --model gemini

# Query using local embeddings (default)
python -m src.main --query "late night studying"

# Query using Gemini embeddings
python -m src.main --query "late night studying" --model gemini

# Rule-based mode (no vector store needed)
python -m src.main --mode mood_first
```

`--query` and `--mode` are independent paths. `--query` uses the RAG pipeline; `--mode` uses the original rule-based scorer. The `--model` flag only applies to RAG paths (`--query`, `--build`).

---

## Sample Interactions

_At least 2–3 examples of inputs and resulting AI outputs — to be filled in after Milestone 2._

---

## Design Decisions

**Module structure split by RAG stage (`ingestion/`, `retrieval/`, `generation/`, `models/`)**
The original flat `src/` had `rag_pipeline.py` and `rag_retriever.py` doing too much in one place, so we split by pipeline responsibility, each package owns one stage. The **tradeoff** is more files to navigate, but each module is now independently testable and easier to extend without touching unrelated logic.

**Embedding backend abstraction (`models/base.py`)**
Local and Gemini embeddings implement the same interface, so the pipeline never cares which backend is active — swapping is a flag, not a code change. The cost is one extra layer of indirection that adds little value while the backend count stays at two.

**`--query` and `--mode` as independent CLI paths**
RAG and rule-based scoring run separate routes from `main.py` instead of being merged. This preserved the original scorer behavior without regression risk, at the cost of maintaining two code paths long-term.

## Resources
- [Gemini Models](https://ai.google.dev/gemini-api/docs/embeddings?_gl=1*qmhatj*_up*MQ..&gclid=Cj0KCQjw77bPBhC_ARIsAGAjjV_LSD67OK3a-1I0dsf37Q4sJvFl5i2-J1AgK1BgLVioQQFFVDX0xQQaAvulEALw_wcB&gbraid=0AAAAACn9t664QWmBEIdkoR3D6rHForFbQ#gemini-embedding)