# API Contract — Redrob Candidate Ranking System

## Overview

The ranking system exposes a single logical function: given a job description and a
candidate pool, return a ranked shortlist with evidence-grounded reasoning.

---

## Input Schema

### Job Description
```json
{
  "job_description": "string (required) — full text of the job description"
}
```

### Candidate Record (from candidates.jsonl)
```json
{
  "candidate_id": "string — format CAND_XXXXXXX",
  "profile": {
    "current_title": "string",
    "current_company": "string",
    "years_of_experience": "number (0–50)",
    "location": "string",
    "summary": "string"
  },
  "career_history": [
    {
      "title": "string",
      "company": "string",
      "duration_months": "integer",
      "description": "string — actual work done (primary signal used for ranking)"
    }
  ],
  "skills": [
    {
      "name": "string",
      "proficiency": "beginner | intermediate | advanced | expert",
      "duration_months": "integer"
    }
  ],
  "redrob_signals": {
    "open_to_work_flag": "boolean",
    "recruiter_response_rate": "number (0.0–1.0)",
    "last_active_date": "date string YYYY-MM-DD",
    "notice_period_days": "integer",
    "github_activity_score": "number (-1 = not linked, 0–100)"
  }
}
```

---

## Output Schema

### submission.csv
```
candidate_id, rank, score, reasoning
```

| Field | Type | Description |
|---|---|---|
| candidate_id | string | Unique identifier matching input |
| rank | integer | 1 = best fit, 100 = 100th best |
| score | float (4 dp) | Composite score (0.0–1.0, higher = better fit) |
| reasoning | string (max 350 chars) | Evidence quote from career history + availability signal |

### Example Output Row
```
CAND_0046525, 1, 0.7879,
"Evidence (Senior ML Engineer @ LinkedIn): 'The architecture combined BM25 +
dense retrieval (BGE embeddings, FAISS HNSW) with an LLM-based re-ranker...'
| Active & responsive on platform"
```

---

## Score Composition

```
final_score = (0.40 × semantic_fit)
            + (0.25 × hard_requirement_score)
            + (0.20 × behavioral_score)
            + (0.10 × location_score)
            - (0.50 × disqualifier_penalty)
            - (0.30 × red_flag_penalty)
```

All component scores are in range [0.0, 1.0]. Final score is clipped to minimum 0.

---

## Error & Edge Case Handling

| Scenario | Handling |
|---|---|
| Missing career history | Semantic score based on profile summary only; hard_req = 0 |
| Missing redrob_signals | behavioral_score = 0 (no penalty applied) |
| Expert skill + 0 duration | Candidate flagged as honeypot, excluded from ranking |
| Experience years vs career history mismatch > 5 yrs | Flagged as honeypot, excluded |
| Candidate outside India | location_score = 0 (no penalty, just no bonus) |
| Equal scores (tie) | Ordered by candidate_id ascending |

---

## Constraints

| Constraint | Value |
|---|---|
| Max candidates | 100,000 |
| Ranking runtime (Stage 2) | Under 5 minutes, CPU only, 16GB RAM |
| LLM API calls at rank time | Zero |
| Output size | Exactly 100 candidates |
| GPU required | No |
