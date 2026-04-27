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


**Pytest**

![pytest](/assets/pytest_rag.gif)
---

## Design Decisions

**Module structure split by RAG stage (`ingestion/`, `retrieval/`, `generation/`, `models/`)**
The original flat `src/` had `rag_pipeline.py` and `rag_retriever.py` doing too much in one place, so we split by pipeline responsibility, each package owns one stage. The **tradeoff** is more files to navigate, but each module is now independently testable and easier to extend without touching unrelated logic.

**Embedding backend abstraction (`models/base.py`)**
Local and Gemini embeddings implement the same interface, so the pipeline never cares which backend is active — swapping is a flag, not a code change. The cost is one extra layer of indirection that adds little value while the backend count stays at two.

**`--query` and `--mode` as independent CLI paths**
RAG and rule-based scoring run separate routes from `main.py` instead of being merged. This preserved the original scorer behavior without regression risk, at the cost of maintaining two code paths long-term.

## Reliability and Evaluation

### Automated Tests
- `tests/test_recommender.py` — existing unit tests (unchanged)
- `tests/test_rag_retriever.py` — semantic retrieval correctness (`src/retrieval/retriever.py`)
- `tests/test_rag_pipeline.py` — confidence flag trigger logic (`src/generation/pipeline.py`)

### Confidence Scoring
After retrieval, `pipeline.py` reads the similarity score of the top results. `low_confidence` will be triggered if set to `True` and its below `CONFIDENCE_THRESHOLD = 0.5`. With only 20 songs, even correct semantic matches in the 0.35-0.5 range, so it is excepted for most queries and signals due to catalog limitations rather than a model failure.

### Logging and Error Handling
`src/logger.py` appends structured entries to `melody_cli.log` after every session. Each entry includes: ISO timestamp, path (`query` or `mode`), query text or mode name, list of retrieved song titles with scores, the `low_confidence` flag, and any warnings.

**Missing Vector Store**
![log_vector](/assets/logger_vector.gif)

Caught in the `main.py`, it is written to the log with full traceback in the `log_exception()`, preventing the app to never crash silently.

### Human Evaluation

Query 1: "songs to listen to when it's raining"
![rain](/assets/human_emotion.gif)

Rainy days wants me to listen to relaxing or melancholic songs, but it seems that the model added a more intense song in the list. I can see that it was added might be cause of the title of the song "Storm Runner".

Query 2: "background music for a dinner party"
![background](/assets/human_activity.gif)

Might be cause the lack of diverse of genres that it will pick the same few genres since they are within the same range. Adding more will help reduce those.

Query 3: "Vibes"
![dreamy](/assets/human_vibe.gif)

This query output seems like the closest that it retrieved for the song recommendations. A mix of music that fits the vibe.

---

## Reflection and Ethics

### Limitations and Biases

**Catalog size:** Only 20 songs in the csv, resulting into a narrow range for cosine similarity scores and triggering LOW CONFIDENCE warning, even when correct. Having more songs in the data would provide a meaningful threshold and output.

**Genre/mood coverage gaps:** Catalog has missing genres that will receive irrelevant results (what we need to a few test cases); this results to system warnings rather than silently returning bad results.

**Local embedding model quality:** `all-MiniLM-L6-v2` is a general-purpose sentence transformer not fine-tuned on music descriptions. It understands "chill lofi studying" but may miss genre-specific vocabulary ("drop", "808s", "4/4 time"). A music-domain fine-tuned model would improve retrieval precision.

**Score scale mismatch:** Rule-based scores (0–~4) and RAG similarity scores (0–1) use the same "Score" column label but represent different things. A user comparing both modes' outputs could be misled.

### Misuse Potential

Currently it is very minimal for misuse risk. A theoretical concern is that a biased catalog (e.g., over-representing one demographic's music) could systematically under-serve users whose tastes don't match the catalog. Mitigation: catalog curation should be intentional and diverse. The confidence warning already tells users when the catalog fails them.

### What Surprised Me During Testing

**RAG and rule-based agree more than expected:** 3 out of 5 profiles for both modes returned the same song at first place. The semantic model has learned enough about music mood/genre labels that it ranks similarity to explicit categorical matching... only for the top position...

**Low confidence on correct results:** The lofi query returned three songs correctly only to be triggered by the LOW CONFIDENCE at 0.47. Threshold is needed to be calibrated for large catalogs. While the warning is technically accurate but misleading, the retrieval was correct.

### AI Collaboration

**Helpful:** AI assistance was valuable during the refactor from `rag_pipeline.py` into the `src/generation/`, `src/retrieval/`, `src/ingestion/`, `src/models/` package structure. The decomposition of concerns (retriever vs. pipeline vs. model backend) was designed collaboratively, and the resulting structure made the confidence scoring and model-swapping logic easier to test in isolation.

**Flawed:** During early implementation, the AI suggested using `np.dot` for cosine similarity without first normalising the query vector. This produced incorrect scores for queries where the embedding magnitude wasn't 1.0. The fix was caught during manual testing with adversarial queries that should have scored near zero but didn't.

Also implemented wrong models that were no longer available to use. This resulted with reading documentations from Gemini API Docs to provide the context and changes needed to be made.

## Resources
- [Gemini Models](https://ai.google.dev/gemini-api/docs/embeddings?_gl=1*qmhatj*_up*MQ..&gclid=Cj0KCQjw77bPBhC_ARIsAGAjjV_LSD67OK3a-1I0dsf37Q4sJvFl5i2-J1AgK1BgLVioQQFFVDX0xQQaAvulEALw_wcB&gbraid=0AAAAACn9t664QWmBEIdkoR3D6rHForFbQ#gemini-embedding)