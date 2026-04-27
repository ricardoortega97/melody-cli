## MelodyCLI

**Project name:** ai110-music-rec-sim (now melodyCLI)

ai110-music-rec-sim was a CLI music recommender that scored each song in a catalog against a user's stated taste profile — genre, mood, and audio features (energy, tempo, valence, acousticness) — using a point-weight system. Songs were filtered for eligibility, scored across three signals (genre match +2.0, mood match +1.0, vibe closeness up to +1.5), then ranked and returned as a top-K list. A session feedback loop tracked skips and dislikes, applying cooldowns and disqualifications that narrowed future recommendations over time.

---

## Title and Summary

**melodyCLI — RAG-Powered Music Recommendation CLI**

melodyCLI upgrades the original rule-based recommender with a RAG layer so users can describe what they want in plain language instead of picking genre and mood labels. A natural language query is embedded and matched against the song catalog via semantic vector search; the top-k semantic candidates are then passed to the existing scoring engine, which applies audio-feature ranking on top of the semantic retrieval. The result is a CLI that accepts inputs like `"late night studying"` or `"high energy gym warmup"` and returns ranked, explained recommendations — with a confidence warning when the query falls outside the catalog's coverage.

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

_Step-by-step directions to run the code — to be written after implementation._

---

## Sample Interactions

_At least 2–3 examples of inputs and resulting AI outputs — to be filled in after Milestone 2._

---

## Design Decisions

_Why you built it this way and what trade-offs you made — to be written._