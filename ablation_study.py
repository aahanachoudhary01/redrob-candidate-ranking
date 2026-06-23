"""
Ablation Study — Redrob Candidate Ranking System
--------------------------------------------------
Compares 4 system configurations to show each component's contribution:

  Config 1: Keyword matching only (baseline — what most systems do)
  Config 2: Semantic similarity only (embeddings, no rules)
  Config 3: Semantic + Hard requirement scoring
  Config 4: Full system (semantic + hard req + behavioral + location + penalties)

How to run:
    python ablation_study.py

Output: ablation_results.csv + printed comparison table
"""

import json
import numpy as np
import csv
from datetime import datetime
from sentence_transformers import SentenceTransformer

# ── Load data ─────────────────────────────────────────────────────────────
print("Loading data...")
with open("candidate_ids.json")   as f: candidate_ids = json.load(f)
with open("candidate_lookup.json") as f: lookup = json.load(f)
embeddings = np.load("candidate_embeddings.npy")

JOB_DESCRIPTION = """
Senior AI Engineer at Redrob AI. 5-9 years experience.
Production experience with embeddings-based retrieval systems
(sentence-transformers, BGE, E5) deployed to real users.
Production experience with vector databases: Pinecone, Weaviate, FAISS, Elasticsearch.
Strong Python. Evaluation frameworks: NDCG, MRR, MAP, A/B testing.
Ideal: shipped end-to-end ranking or search system at a product company.
"""

# Known strong-fit terms (used as ground-truth proxy for ablation scoring)
GROUND_TRUTH_SIGNALS = [
    "faiss", "pinecone", "weaviate", "elasticsearch", "vector",
    "embedding", "bge", "sentence-transformer", "ndcg", "mrr",
    "ranking", "retrieval", "learning to rank", "a/b test", "hybrid search",
]

HARD_REQUIREMENT_TERMS = [
    "embedding", "sentence-transformer", "bge", "dense retrieval",
    "semantic search", "retrieval", "pinecone", "weaviate", "faiss",
    "vector database", "vector store", "hybrid search", "nearest neighbor",
    "hnsw", "bm25", "ranking system", "re-ranker", "ndcg", "mrr",
    "a/b test", "evaluation framework", "learning to rank", "lambdamart",
    "elasticsearch", "opensearch", "qdrant", "milvus",
]

CONSULTING_COMPANIES = ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"]
CV_SPEECH_TERMS = ["computer vision", "speech recognition", "object detection", "image segmentation", "asr", "tts"]
NLP_IR_TERMS = ["nlp", "natural language", "information retrieval", "search", "ranking", "text"]
PREFERRED_LOCATIONS = ["noida", "pune", "hyderabad", "mumbai", "delhi", "ncr", "gurgaon"]

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
jd_vec = model.encode([JOB_DESCRIPTION])[0]

# ── Cosine similarity ─────────────────────────────────────────────────────
jd_norm   = jd_vec / np.linalg.norm(jd_vec)
cand_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
sem_scores = (cand_norm @ jd_norm).tolist()

# ── Helper functions ──────────────────────────────────────────────────────
def full_text(c):
    parts = [c["profile"].get("summary","")]
    for j in c.get("career_history",[]):
        parts += [j.get("description",""), j.get("title",""), j.get("company","")]
    return " ".join(parts).lower()

def keyword_score(c):
    """Config 1: pure keyword matching against JD terms."""
    text = full_text(c)
    skills = [s["name"].lower() for s in c.get("skills",[])]
    all_text = text + " " + " ".join(skills)
    hits = sum(1 for t in GROUND_TRUTH_SIGNALS if t in all_text)
    return min(hits / 6.0, 1.0)

def hard_req_score(c):
    text = full_text(c)
    hits = sum(1 for t in HARD_REQUIREMENT_TERMS if t in text)
    return min(hits / 4.0, 1.0)

def behavioral_score(c):
    sig = c.get("redrob_signals",{})
    score = 0.0
    last_active = sig.get("last_active_date","")
    if last_active:
        try:
            days = (datetime.now() - datetime.strptime(last_active, "%Y-%m-%d")).days
            if days <= 14:   score += 0.4
            elif days <= 30: score += 0.3
            elif days <= 90: score += 0.15
        except: pass
    score += 0.3 * sig.get("recruiter_response_rate", 0)
    if sig.get("open_to_work_flag"): score += 0.15
    notice = sig.get("notice_period_days", 60)
    if notice <= 30:  score += 0.15
    elif notice <= 60: score += 0.07
    return min(score, 1.0)

def location_score(c):
    loc = c["profile"].get("location","").lower()
    return 1.0 if any(p in loc for p in PREFERRED_LOCATIONS) else 0.3

def penalty_score(c):
    text = full_text(c)
    penalty = 0.0
    # research without production
    is_research = any(t in text for t in ["research scientist","academic","phd researcher"])
    has_prod    = any(t in text for t in ["production","deployed","shipped","users"])
    if is_research and not has_prod: penalty += 0.5
    # pure consulting
    history  = c.get("career_history",[])
    all_cos  = [j.get("company","").lower() for j in history] + [c["profile"].get("current_company","").lower()]
    con_hits = sum(1 for co in all_cos if any(cc in co for cc in CONSULTING_COMPANIES))
    if con_hits > 0 and con_hits == len(all_cos) and len(history) >= 2:
        penalty += 0.4
    # wrong domain
    has_cv  = any(t in text for t in CV_SPEECH_TERMS)
    has_nlp = any(t in text for t in NLP_IR_TERMS)
    if has_cv and not has_nlp: penalty += 0.35
    # title chaser
    short = sum(1 for j in history if j.get("duration_months",0) < 18)
    if len(history) >= 4 and short >= 3: penalty += 0.25
    return min(penalty, 1.0)

def is_honeypot(c):
    for s in c.get("skills",[]):
        if s.get("proficiency") == "expert" and s.get("duration_months", 1) == 0:
            return True
    total = sum(j.get("duration_months",0) for j in c.get("career_history",[]))
    if abs((total/12) - c["profile"].get("years_of_experience",0)) > 5:
        return True
    return False

# ── Run 4 configs ─────────────────────────────────────────────────────────
print("Running ablation across 4 configurations...")

configs = {
    "Config 1: Keyword Only":         lambda i,c: keyword_score(c),
    "Config 2: Semantic Only":        lambda i,c: float(sem_scores[i]),
    "Config 3: Semantic + Hard Req":  lambda i,c: 0.60*float(sem_scores[i]) + 0.40*hard_req_score(c),
    "Config 4: Full System":          lambda i,c: max(
        0.40*float(sem_scores[i]) + 0.25*hard_req_score(c) +
        0.20*behavioral_score(c) + 0.10*location_score(c) - 0.5*penalty_score(c), 0),
}

results = {}
for name, fn in configs.items():
    ranked = []
    for i, cid in enumerate(candidate_ids):
        c = lookup[cid]
        if is_honeypot(c): continue
        ranked.append((cid, fn(i, c)))
    ranked.sort(key=lambda x: -x[1])
    results[name] = [cid for cid, _ in ranked[:100]]

# ── Pseudo-NDCG@10 ───────────────────────────────────────────────────────
# Ground truth: candidates whose career history mentions 3+ hard req terms
# (used as proxy relevance labels since no official ground truth provided)
def is_relevant(cid):
    c = lookup[cid]
    text = full_text(c)
    hits = sum(1 for t in HARD_REQUIREMENT_TERMS if t in text)
    return hits >= 3

def ndcg_at_k(ranked_list, k=10):
    dcg = sum(
        (1 if is_relevant(cid) else 0) / np.log2(i+2)
        for i, cid in enumerate(ranked_list[:k])
    )
    ideal = sum(1/np.log2(i+2) for i in range(k))
    return round(dcg / ideal, 4) if ideal > 0 else 0

def precision_at_k(ranked_list, k=10):
    return round(sum(1 for cid in ranked_list[:k] if is_relevant(cid)) / k, 4)

def mean_avg_precision(ranked_list):
    hits, ap_sum = 0, 0.0
    for i, cid in enumerate(ranked_list[:100]):
        if is_relevant(cid):
            hits += 1
            ap_sum += hits / (i+1)
    return round(ap_sum / max(hits, 1), 4)

# ── Print results ─────────────────────────────────────────────────────────
print("\n" + "="*70)
print(f"{'Configuration':<40} {'NDCG@10':>8} {'P@10':>8} {'MAP@100':>8}")
print("="*70)

eval_rows = []
for name, top100 in results.items():
    n10 = ndcg_at_k(top100, 10)
    p10 = precision_at_k(top100, 10)
    mp  = mean_avg_precision(top100)
    print(f"{name:<40} {n10:>8} {p10:>8} {mp:>8}")
    eval_rows.append({"configuration": name, "NDCG@10": n10, "P@10": p10, "MAP@100": mp})

print("="*70)
print("\nInterpretation:")
print("  NDCG@10  — ranking quality of top 10 (1.0 = perfect order)")
print("  P@10     — fraction of top 10 that are genuinely relevant")
print("  MAP@100  — mean average precision across top 100")
print("\nNote: relevance labels are proxy-based (3+ hard req term matches")
print("in career history). No official ground truth provided in dataset.")

with open("ablation_results.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=["configuration","NDCG@10","P@10","MAP@100"])
    w.writeheader(); w.writerows(eval_rows)

print("\n✅ ablation_results.csv saved.")
