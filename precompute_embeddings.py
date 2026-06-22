"""
STEP 1: Precompute candidate embeddings
-----------------------------------------
This runs ONCE (can take 10-20 minutes, that's fine - it's an offline step,
NOT part of the 5-minute ranking constraint).

What it does:
- Reads all 100,000 candidates
- Builds a "career text" for each candidate (their actual work history,
  NOT just their skills list - this is critical, since the JD explicitly
  says skills-keyword-matching is a trap)
- Converts that text into embeddings using a local model (no API calls)
- Saves embeddings + candidate IDs to disk so the ranking script can
  load them instantly later

How to run:
    python precompute_embeddings.py

Output files created:
    candidate_embeddings.npy   -> the embedding vectors
    candidate_ids.json         -> matching candidate IDs (same order)
    candidate_lookup.json      -> full candidate data, indexed by ID (for fast lookup later)
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime

DATA_FILE = "candidates.jsonl"

print("Loading candidates...")
candidates = []
with open(DATA_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            candidates.append(json.loads(line))
print(f"Loaded {len(candidates)} candidates.")


def build_career_text(candidate):
    """
    Build a text blob that represents what the candidate has actually DONE,
    not just what skills they've listed. This matters because the JD
    explicitly warns: a candidate with all the right keywords in their
    skills list but no real career evidence is a trap.
    """
    profile = candidate["profile"]
    parts = [
        f"Current role: {profile.get('current_title','')} at {profile.get('current_company','')} "
        f"in {profile.get('current_industry','')} industry.",
        f"Summary: {profile.get('summary','')}",
    ]

    for job in candidate.get("career_history", []):
        parts.append(
            f"Worked as {job.get('title','')} at {job.get('company','')} "
            f"({job.get('industry','')}, company size {job.get('company_size','')}) "
            f"for {job.get('duration_months',0)} months. "
            f"Description: {job.get('description','')}"
        )

    # Include skill names too, but they carry less weight since they're
    # mixed in with a lot more career-history text (de-prioritized naturally)
    skills = ", ".join(s["name"] for s in candidate.get("skills", []))
    parts.append(f"Listed skills: {skills}")

    return " ".join(parts)


print("Building career text for each candidate...")
career_texts = [build_career_text(c) for c in candidates]
candidate_ids = [c["candidate_id"] for c in candidates]

print("Loading embedding model (first run will download ~80MB model)...")
# all-MiniLM-L6-v2 is small, fast, and runs well on CPU - good for this use case
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Encoding all candidates... this is the slow part, please wait (10-20 min)")
embeddings = model.encode(
    career_texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
)

print(f"Done. Embeddings shape: {embeddings.shape}")

# Save embeddings
np.save("candidate_embeddings.npy", embeddings)

# Save candidate IDs in the same order as embeddings
with open("candidate_ids.json", "w", encoding="utf-8") as f:
    json.dump(candidate_ids, f)

# Save full candidate data indexed by ID for fast lookup during ranking
lookup = {c["candidate_id"]: c for c in candidates}
with open("candidate_lookup.json", "w", encoding="utf-8") as f:
    json.dump(lookup, f)

print("\n✅ All done!")
print("Files created: candidate_embeddings.npy, candidate_ids.json, candidate_lookup.json")
print("You only need to run this script ONCE. Now run rank_candidates.py")
