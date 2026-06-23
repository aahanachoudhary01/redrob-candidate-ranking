# Redrob Candidate Ranking System
India.Runs Hackathon — Data & AI Challenge

---

## The Problem I Set Out to Solve

When I read the JD, one line stuck with me: *"Not by matching keywords, but by actually understanding who fits."*

Most systems would embed the JD, embed the candidates, run cosine similarity, and call it done. The problem is that approach still fundamentally rewards whoever wrote the right words in their skills section — which is exactly the trap the challenge warned about.

So I built something different.

---

## My Approach

The core idea is simple: **rank candidates based on what they have actually done, not what they have listed.**

Instead of embedding a candidate's skills list, I build a "career narrative" from their actual job descriptions — the real work they describe doing, the systems they built, the scale they operated at. That narrative gets embedded and compared against the JD.

This means a candidate who spent three years building hybrid retrieval systems but never wrote "vector database" in their skills section still surfaces near the top. And a candidate who has every keyword in their profile but whose actual job descriptions say nothing technical gets pushed down.

The ranking runs in two stages. First, a one-time precompute step that converts all 100,000 career narratives into embeddings using a local model (no API calls, runs offline). Then a fast ranking step that loads those embeddings, compares them against the JD, applies scoring rules, and produces the final output — all under two minutes on a standard CPU.

---

## How Scoring Works

Semantic fit between the JD and the candidate's career narrative carries the most weight at 40%. This is the core signal — does this person's actual work history match what the role needs?

Hard requirement matching accounts for 25%. I check whether the candidate's career history contains real evidence of working with embeddings-based retrieval, vector infrastructure (FAISS, Pinecone, Weaviate, etc.), and evaluation frameworks like NDCG or A/B testing. Importantly, I check across a wide set of synonyms and related terms — not just exact strings — so a candidate who wrote "nearest neighbor index" instead of "FAISS" still gets credit.

Behavioral availability is 20%. The JD was explicit: a perfect-on-paper candidate who hasn't been active in six months and has a 5% recruiter response rate is not actually available. I factor in last active date, recruiter response rate, open-to-work status, and notice period.

Location fit is the remaining 10%, with preference for Noida, Pune, Delhi NCR, Hyderabad, and Mumbai as the JD specified.

Penalties are subtracted rather than multiplied. A single red flag shouldn't eliminate someone who is otherwise a strong match — it should reduce their score proportionally.

---

## Architecture

candidates.jsonl (100K records)
        ↓
Career Narrative Builder (job descriptions, not skills list)
        ↓
all-MiniLM-L6-v2 (local model, no API)
        ↓
Saved Embeddings ──────────────────────────────┐
                                               ↓

                                               
Job Description → Embedded → Cosine Similarity
                                               ↓
                                               
                              Rule-based Scoring
                         (hard req + behavioral + penalties)
                                               ↓
                               Top 100 with Evidence Quotes

## Traps I Specifically Handled

The dataset has honeypot profiles — candidates with impossible combinations like "expert" proficiency in a skill with zero months of experience, or claimed experience years that don't add up against their career history. These are filtered out entirely before ranking.

Beyond honeypots, I penalize candidates whose entire career history is at pure services firms with no product company experience, candidates showing clear title-chasing patterns (multiple roles under 18 months each), and candidates with computer vision or speech backgrounds but no NLP or information retrieval exposure.

Out of 100,000 candidates, 60 were identified as honeypots and excluded.

---

## What the Output Looks Like

Each of the top 100 candidates comes with a reasoning line that quotes directly from their career history — not a template phrase, but an actual sentence from their job description that proves why they ranked where they did. A recruiter can read rank 1 and immediately see the specific system this person built and where.

---

## How to Run It

Install dependencies:
```
pip install pandas sentence-transformers scikit-learn numpy
```

Run precompute once (takes 15-20 minutes, only needed once):
```
python precompute_embeddings.py
```

Run the ranker (under 2 minutes):
```
python rank_candidates.py
```

Validate the output:
```
python validate_submission.py submission.csv
```

---

## Files in This Repo

precompute_embeddings.py — builds and saves candidate embeddings (one-time step)

rank_candidates.py — the main ranker, runs within the 5-minute CPU constraint

explore_data.py — data exploration and trap pattern analysis

verify_top_candidates.py — prints full career details of top N candidates for manual review

submission.csv — final ranked output, top 100 candidates

---

## Stack

Python, sentence-transformers (all-MiniLM-L6-v2), NumPy, pandas, scikit-learn. No GPU. No external API calls at ranking time.
