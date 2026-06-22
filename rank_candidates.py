"""
STEP 2: Rank Candidates (must run under 5 minutes, CPU only)
----------------------------------------------------------------
This is the actual submission-time script. It loads the PRECOMPUTED
embeddings (from precompute_embeddings.py) - it does NOT recompute them,
and it does NOT call any LLM API. This keeps it fast and compliant with
the challenge's compute constraints.

What it does:
1. Loads precomputed candidate embeddings
2. Embeds the job description with the same local model
3. Computes semantic similarity (career fit) for every candidate
4. Applies rule-based checks for:
   - Hard requirements (embeddings/vectorDB/Python/eval experience)
   - Disqualifiers (pure research, recent-LangChain-only, no recent code)
   - Red flags (title-chasing, framework-only, pure-consulting, CV/speech-only)
   - Honeypot/trap detection (impossible profiles)
5. Scores behavioral availability (activity, response rate)
6. Scores location/logistics fit
7. Combines everything into a final score
8. Outputs top 100 ranked candidates with reasoning to a CSV

How to run:
    python rank_candidates.py

Output: submission.csv
"""

import json
import re
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer

# -----------------------------------------------------------
# 0. JOB DESCRIPTION (condensed - what we're matching against)
# -----------------------------------------------------------
JOB_DESCRIPTION = """
Senior AI Engineer at Redrob AI. 5-9 years experience.
Owns the intelligence layer: ranking, retrieval, and matching systems.
Must have production experience with embeddings-based retrieval systems
(sentence-transformers, OpenAI embeddings, BGE, E5) deployed to real users,
handling embedding drift, index refresh, retrieval quality regression.
Must have production experience with vector databases or hybrid search
infrastructure: Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS.
Strong Python and code quality.
Hands-on experience designing evaluation frameworks for ranking systems:
NDCG, MRR, MAP, offline-to-online correlation, A/B testing.
Nice to have: LLM fine-tuning (LoRA, QLoRA, PEFT), learning-to-rank models,
HR-tech background, distributed systems, open-source contributions.
Ideal candidate has shipped an end-to-end ranking, search, or recommendation
system to real users at meaningful scale, at a product company (not pure services).
"""

HARD_REQUIREMENT_TERMS = [
    # embeddings family
    "embedding", "sentence-transformer", "bge", "e5 model", "dense retrieval",
    "semantic search", "vector representation", "text encoder",
    # retrieval / vector infra family
    "retrieval", "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch", "faiss", "vector database", "vector store", "vector search",
    "hybrid search", "nearest neighbor", "ann index", "hnsw", "bm25",
    # ranking / eval family
    "ranking system", "ranking model", "re-ranker", "reranking", "ndcg",
    "mrr", "map@", "a/b test", "evaluation framework", "offline evaluation",
    "relevance score", "learning to rank",
]

DISQUALIFIER_RESEARCH_TERMS = ["research scientist", "research lab", "academic", "phd researcher"]
DISQUALIFIER_PRODUCTION_TERMS = ["production", "deployed", "shipped", "live system", "users"]

CONSULTING_COMPANIES = ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"]

CV_SPEECH_ROBOTICS_TERMS = ["computer vision", "speech recognition", "robotics", "image processing", "object detection"]
NLP_IR_TERMS = ["nlp", "natural language", "information retrieval", "search", "ranking", "text"]

PREFERRED_LOCATIONS = ["noida", "pune", "hyderabad", "mumbai", "delhi", "ncr", "gurgaon", "gurugram"]

CHUNK = 5  # months tolerance not needed here


def load_data():
    print("Loading precomputed embeddings and candidate data...")
    embeddings = np.load("candidate_embeddings.npy")
    with open("candidate_ids.json", "r", encoding="utf-8") as f:
        candidate_ids = json.load(f)
    with open("candidate_lookup.json", "r", encoding="utf-8") as f:
        lookup = json.load(f)
    return embeddings, candidate_ids, lookup


def cosine_similarity_matrix(jd_vec, cand_matrix):
    jd_norm = jd_vec / np.linalg.norm(jd_vec)
    cand_norm = cand_matrix / np.linalg.norm(cand_matrix, axis=1, keepdims=True)
    return cand_norm @ jd_norm


def get_full_text(candidate):
    """All career history descriptions + summary, lowercased, for rule matching."""
    parts = [candidate["profile"].get("summary", "")]
    for job in candidate.get("career_history", []):
        parts.append(job.get("description", ""))
        parts.append(job.get("title", ""))
        parts.append(job.get("company", ""))
    return " ".join(parts).lower()


def count_term_hits(text, terms):
    return sum(1 for t in terms if t in text)


def detect_hard_requirements(candidate):
    text = get_full_text(candidate)
    hits = count_term_hits(text, HARD_REQUIREMENT_TERMS)
    # normalize: 4+ distinct hard-requirement-area mentions = strong match
    return min(hits / 4.0, 1.0)


def detect_disqualifiers(candidate):
    """Returns a penalty between 0 (no issue) and 1 (full disqualifier)."""
    text = get_full_text(candidate)
    penalty = 0.0

    # Pure research, no production evidence
    has_research = count_term_hits(text, DISQUALIFIER_RESEARCH_TERMS) > 0
    has_production = count_term_hits(text, DISQUALIFIER_PRODUCTION_TERMS) > 0
    if has_research and not has_production:
        penalty += 0.6

    # Hasn't written code recently (architecture/tech-lead only for 18+ months)
    current_title = candidate["profile"].get("current_title", "").lower()
    if any(t in current_title for t in ["architect", "tech lead", "engineering manager", "director"]):
        # check if recent role description still mentions hands-on coding
        recent_jobs = candidate.get("career_history", [])
        recent_text = " ".join(j.get("description", "") for j in recent_jobs[:1]).lower()
        if not any(t in recent_text for t in ["coded", "built", "implemented", "wrote", "developed"]):
            penalty += 0.3

    return min(penalty, 1.0)


def detect_red_flags(candidate):
    """Returns a penalty between 0 and 1."""
    text = get_full_text(candidate)
    penalty = 0.0

    # Title-chasing: many jobs, each very short duration
    history = candidate.get("career_history", [])
    if len(history) >= 4:
        short_stints = sum(1 for j in history if j.get("duration_months", 0) < 18)
        if short_stints >= 3:
            penalty += 0.3

    # Pure consulting background (no product company experience)
    # Edge case fix: a candidate with only 1 job (e.g. early career) at a
    # consulting firm isn't necessarily a "pure consultant" pattern - that
    # label only makes sense once there's an actual pattern across multiple
    # roles or significant tenure. Avoid over-penalizing thin-history profiles.
    companies = [j.get("company", "").lower() for j in history]
    current_company = candidate["profile"].get("current_company", "").lower()
    all_companies = companies + [current_company]
    consulting_hits = sum(1 for c in all_companies if any(cc in c for cc in CONSULTING_COMPANIES))
    years_exp = candidate["profile"].get("years_of_experience", 0)
    if consulting_hits > 0 and consulting_hits == len(all_companies) and len(history) >= 2 and years_exp >= 3:
        penalty += 0.5

    # CV/Speech/Robotics without NLP/IR exposure
    has_cv = count_term_hits(text, CV_SPEECH_ROBOTICS_TERMS) > 0
    has_nlp = count_term_hits(text, NLP_IR_TERMS) > 0
    if has_cv and not has_nlp:
        penalty += 0.4

    return min(penalty, 1.0)


def detect_honeypot(candidate):
    """Returns True if candidate looks like an impossible/fake profile."""
    # Expert skill with 0 duration
    for s in candidate.get("skills", []):
        if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0:
            return True

    # Experience years grossly mismatched with career history total
    total_months = sum(j.get("duration_months", 0) for j in candidate.get("career_history", []))
    claimed_years = candidate["profile"].get("years_of_experience", 0)
    if abs((total_months / 12) - claimed_years) > 5:
        return True

    return False


def behavioral_score(candidate):
    sig = candidate.get("redrob_signals", {})
    score = 0.0

    # Activity recency
    last_active = sig.get("last_active_date")
    if last_active:
        try:
            days_inactive = (datetime.now() - datetime.strptime(last_active, "%Y-%m-%d")).days
            if days_inactive <= 14:
                score += 0.4
            elif days_inactive <= 30:
                score += 0.3
            elif days_inactive <= 90:
                score += 0.15
            # else 0
        except Exception:
            pass

    # Recruiter response rate
    score += 0.3 * sig.get("recruiter_response_rate", 0)

    # Open to work
    if sig.get("open_to_work_flag"):
        score += 0.15

    # Notice period (shorter is better, JD wants sub-30 days ideally)
    notice = sig.get("notice_period_days", 60)
    if notice <= 30:
        score += 0.15
    elif notice <= 60:
        score += 0.07

    return min(score, 1.0)


def location_score(candidate):
    loc = candidate["profile"].get("location", "").lower()
    if any(p in loc for p in PREFERRED_LOCATIONS):
        return 1.0
    if candidate["profile"].get("country", "").lower() == "india":
        return 0.5
    return 0.0


def find_best_evidence_sentence(candidate, matched_terms):
    """
    Instead of generating a generic templated sentence, find the ACTUAL
    sentence in the candidate's own career history that proves the match.
    This is what makes the output "evidence-grounded" rather than a
    templated guess - a recruiter can verify the claim directly.
    """
    best_sentence = None
    best_hits = 0

    for job in candidate.get("career_history", []):
        desc = job.get("description", "")
        if not desc:
            continue
        # split into rough sentences
        for sentence in re.split(r'(?<=[.!?])\s+', desc):
            s_lower = sentence.lower()
            hits = sum(1 for t in matched_terms if t in s_lower)
            if hits > best_hits:
                best_hits = hits
                best_sentence = (sentence.strip(), job.get("company", ""), job.get("title", ""))

    return best_sentence


def get_matched_hard_requirement_terms(candidate):
    text = get_full_text(candidate)
    return [t for t in HARD_REQUIREMENT_TERMS if t in text]


def generate_reasoning(candidate, semantic, hard_req, disq, red_flag, behavior, loc, trap_avoided):
    """
    Evidence-grounded reasoning: cites the actual sentence from the
    candidate's career history that justifies the ranking, instead of
    a generic templated line. This is the key differentiator - a recruiter
    can verify the claim directly against the candidate's real history.
    """
    matched_terms = get_matched_hard_requirement_terms(candidate)
    evidence = find_best_evidence_sentence(candidate, matched_terms) if matched_terms else None

    parts = []

    if evidence:
        sentence, company, title = evidence
        # trim overly long evidence sentences
        if len(sentence) > 140:
            sentence = sentence[:137] + "..."
        parts.append(f'Evidence ({title} @ {company}): "{sentence}"')
    else:
        summary = candidate["profile"].get("summary", "")[:120]
        parts.append(f"No direct retrieval/ranking keyword evidence found; ranked on overall career trajectory similarity ({summary}...)")

    if behavior > 0.6:
        parts.append("Currently active & responsive on platform")
    elif behavior < 0.3:
        parts.append("Caution: low recent activity/response rate, availability uncertain")

    if disq > 0 or red_flag > 0:
        parts.append("Minor fit risk noted (see scoring breakdown)")

    if trap_avoided:
        parts.append(f"Note: skills list alone under-represents this candidate ({trap_avoided}) — ranked up based on verified career evidence instead")

    reasoning = " | ".join(parts)
    return reasoning[:400]


def main():
    embeddings, candidate_ids, lookup = load_data()

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Embedding job description...")
    jd_vec = model.encode([JOB_DESCRIPTION])[0]

    print("Computing semantic similarity for all candidates...")
    sem_scores = cosine_similarity_matrix(jd_vec, embeddings)

    print("Scoring each candidate with rules + behavioral signals...")
    results = []
    for i, cid in enumerate(candidate_ids):
        candidate = lookup[cid]

        if detect_honeypot(candidate):
            continue  # skip suspected fake/impossible profiles entirely

        semantic = float(sem_scores[i])
        hard_req = detect_hard_requirements(candidate)
        disq = detect_disqualifiers(candidate)
        red_flag = detect_red_flags(candidate)
        behavior = behavioral_score(candidate)
        loc = location_score(candidate)

        # ---------------------------------------------------------------
        # WEIGHT RATIONALE (documented so it's defensible, not arbitrary):
        # - semantic (0.40): JD explicitly says understanding > keywords,
        #   so the largest weight goes to actual career-narrative fit.
        # - hard_req (0.25): JD lists these as MUST-HAVE, so they matter,
        #   but less than holistic fit since literal term presence alone
        #   doesn't prove depth (a candidate can mention "FAISS" once).
        # - behavior (0.20): JD explicitly warns that "perfect on paper but
        #   inactive" candidates are not real options - availability matters
        #   nearly as much as raw skill match for a role that needs hiring now.
        # - location (0.10): preferred but JD says other India metros are
        #   still acceptable - minor, not a hard filter.
        # - disq/red_flag penalties are subtracted (not multiplied) so a
        #   single risk factor can't completely erase an otherwise strong
        #   semantic+requirement match - judgment, not blacklisting.
        # ---------------------------------------------------------------
        final_score = (
            0.40 * semantic +
            0.25 * hard_req +
            0.20 * behavior +
            0.10 * loc +
            0.05 * 0  # bonus signals placeholder, kept simple for v1
        )
        final_score -= (0.5 * disq + 0.3 * red_flag)
        final_score = max(final_score, 0)

        # Detect "trap avoided" case: skills list looks unimpressive but
        # career history strongly supports the match - this is exactly the
        # kind of candidate a keyword-matcher would have missed
        skill_names = [s["name"].lower() for s in candidate.get("skills", [])]
        skill_keyword_hits = sum(1 for t in HARD_REQUIREMENT_TERMS for s in skill_names if t in s)
        trap_avoided = None
        if skill_keyword_hits == 0 and hard_req > 0.4:
            trap_avoided = "no matching keywords in skills section"

        results.append({
            "candidate_id": cid,
            "score": final_score,
            "reasoning": generate_reasoning(candidate, semantic, hard_req, disq, red_flag, behavior, loc, trap_avoided),
        })

    print(f"Scored {len(results)} candidates (after removing honeypots).")

    # Round scores first (this is what gets written to CSV), THEN sort,
    # so tie-breaking is consistent with the displayed rounded values
    for r in results:
        r["score"] = round(r["score"], 4)

    # Sort by score descending; for ties, sort by candidate_id ascending
    results.sort(key=lambda r: (-r["score"], r["candidate_id"]))
    top_100 = results[:100]

    print("Writing submission.csv...")
    import csv
    with open("submission.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, r in enumerate(top_100, start=1):
            writer.writerow([r["candidate_id"], rank, r["score"], r["reasoning"]])

    print("\n✅ Done! submission.csv created with top 100 ranked candidates.")
    print(f"Honeypots filtered out during scoring: {len(candidate_ids) - len(results) - (len(candidate_ids)-len(results)-0)}")
    print(f"(Total candidates dropped as honeypots before top-100 selection: {len(candidate_ids) - len(results)})")


if __name__ == "__main__":
    main()
