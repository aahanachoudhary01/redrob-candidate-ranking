"""
Manual Verification Script
----------------------------
Purpose: Print full details of top N ranked candidates so you can manually
sanity-check whether the ranking actually makes sense - this is something
judges expect you to have done (not just trust the algorithm blindly).

How to run:
    python verify_top_candidates.py 10
    (10 = how many top candidates to inspect, default 10)
"""

import csv
import json
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10

with open("candidate_lookup.json", "r", encoding="utf-8") as f:
    lookup = json.load(f)

with open("submission.csv", "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print(f"Inspecting top {N} candidates...\n")

for row in rows[:N]:
    cid = row["candidate_id"]
    c = lookup[cid]
    profile = c["profile"]

    print("=" * 70)
    print(f"RANK {row['rank']} | Score: {row['score']} | {cid}")
    print(f"Title: {profile['current_title']} @ {profile['current_company']}")
    print(f"Experience: {profile['years_of_experience']} yrs | Location: {profile['location']}")
    print(f"\nReasoning given: {row['reasoning']}")
    print(f"\nFull career history:")
    for job in c.get("career_history", []):
        print(f"  - {job['title']} @ {job['company']} ({job['duration_months']} months)")
        print(f"    {job['description'][:200]}")
    print()

print("=" * 70)
print(f"\nManually check: do these {N} candidates genuinely look like strong")
print("fits for a Senior AI Engineer (ranking/retrieval) role? Look for:")
print("  - Does the career history actually support the title?")
print("  - Any candidate that looks 'too good' or suspicious (re-check honeypot logic)?")
print("  - Any obviously strong candidate type missing from top ranks?")
