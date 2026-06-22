"""
Day 1-2: Data Exploration Script for Redrob Candidate Ranking Challenge
------------------------------------------------------------------------
What this does:
1. Loads all 100,000 candidates from candidates.jsonl
2. Shows basic stats (experience, skills, locations, etc.)
3. Hunts for "trap" patterns (honeypots, title/description mismatches)

How to run:
1. Make sure candidates.jsonl is in the SAME folder as this script
2. Open Command Prompt in that folder
3. Run: python explore_data.py
"""

import json
from collections import Counter

DATA_FILE = "candidates.jsonl"

print("Loading candidates... (this may take 30-60 seconds for 100k records)")

candidates = []
with open(DATA_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            candidates.append(json.loads(line))

print(f"\n✅ Loaded {len(candidates)} candidates\n")
print("=" * 60)

# -----------------------------------------------------------
# 1. BASIC STATS
# -----------------------------------------------------------
exp_years = [c["profile"]["years_of_experience"] for c in candidates]
print(f"Experience range: {min(exp_years)} - {max(exp_years)} years")
print(f"Average experience: {sum(exp_years)/len(exp_years):.1f} years")

industries = Counter(c["profile"]["current_industry"] for c in candidates)
print(f"\nTop 5 industries:")
for industry, count in industries.most_common(5):
    print(f"  {industry}: {count}")

locations = Counter(c["profile"]["location"] for c in candidates)
print(f"\nTop 5 locations:")
for loc, count in locations.most_common(5):
    print(f"  {loc}: {count}")

# Most common skills across all candidates
all_skills = Counter()
for c in candidates:
    for s in c.get("skills", []):
        all_skills[s["name"]] += 1

print(f"\nTop 10 most common skills:")
for skill, count in all_skills.most_common(10):
    print(f"  {skill}: {count}")

print("\n" + "=" * 60)

# -----------------------------------------------------------
# 2. TRAP DETECTION
# -----------------------------------------------------------
print("\nHUNTING FOR TRAPS...\n")

# Trap A: "expert" skill with 0 months duration (impossible)
trap_a = []
for c in candidates:
    for s in c.get("skills", []):
        if s["proficiency"] == "expert" and s.get("duration_months", 0) == 0:
            trap_a.append(c["candidate_id"])
            break

print(f"Trap A - 'expert' skill with 0 duration_months: {len(trap_a)} candidates")
if trap_a:
    print(f"  Example IDs: {trap_a[:5]}")

# Trap B: years_of_experience doesn't match sum of career_history durations
trap_b = []
for c in candidates:
    total_months = sum(job.get("duration_months", 0) for job in c.get("career_history", []))
    claimed_years = c["profile"]["years_of_experience"]
    # allow some tolerance (overlapping roles, gaps etc.)
    if abs((total_months / 12) - claimed_years) > 5:
        trap_b.append(c["candidate_id"])

print(f"\nTrap B - experience years vs career history mismatch (>5yr gap): {len(trap_b)} candidates")
if trap_b:
    print(f"  Example IDs: {trap_b[:5]}")

# Trap C: candidates with very high skill count but very low github/assessment scores
# (possible keyword-stuffers)
trap_c = []
for c in candidates:
    skill_count = len(c.get("skills", []))
    github_score = c.get("redrob_signals", {}).get("github_activity_score", -1)
    if skill_count > 15 and github_score < 10 and github_score != -1:
        trap_c.append(c["candidate_id"])

print(f"\nTrap C - many skills (15+) but very low github activity (<10): {len(trap_c)} candidates")
if trap_c:
    print(f"  Example IDs: {trap_c[:5]}")

print("\n" + "=" * 60)
print("\nDone! Save this output and share it back to plan the next step.")
