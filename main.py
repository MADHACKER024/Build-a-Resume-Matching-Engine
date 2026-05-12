"""
Redrob AI Campus Hackathon — Resume Matching Engine
=====================================================
Uses ONLY Python standard library (math module).
Implements: Normalization → Dedup → Vocabulary → TF-IDF → JD Binary Vectors → Cosine Similarity
"""

import math

# ──────────────────────────────────────────────────────────────────────────────
# SKILL_ALIASES — provided verbatim; do not modify
# ──────────────────────────────────────────────────────────────────────────────
SKILL_ALIASES = {
    # Languages
    "python": "python",           "pyhton": "python",
    "java": "java",
    "javascript": "javascript",   "javascrpit": "javascript",  "js": "javascript",
    "typescript": "typescript",   "typescrpit": "typescript",
    "c++": "cpp",                 "cpp": "cpp",
    "r": "r",
    "kotlin": "kotlin",
    # ML / Data
    "machinelearning": "machine_learning",  "machine learning": "machine_learning",
    "ml": "machine_learning",               "sklearn": "machine_learning",
    "deeplearning": "deep_learning",        "deep learning": "deep_learning",
    "deep-learning": "deep_learning",
    "tensorflow": "tensorflow",  "pytorch": "pytorch",  "keras": "keras",
    "nlp": "nlp",  "bert": "bert",  "xgboost": "xgboost",
    "feature engineering": "feature_engineering",
    "statistics": "statistics",  "stats": "statistics",
    "regression": "regression",  "clustering": "clustering",
    "data-viz": "data_visualization",       "data visualization": "data_visualization",
    "data viz": "data_visualization",       "matplotlib": "data_visualization",
    "tableau": "data_visualization",        "power-bi": "data_visualization",
    "power bi": "data_visualization",       "powerbi": "data_visualization",
    "pandas": "pandas",  "numpy": "numpy",
    # Web — Frontend
    "react": "react",  "reacts": "react",  "reactjs": "react",
    "vue": "vue",  "vue.js": "vue",  "vuejs": "vue",
    "redux": "redux",  "tailwind": "tailwind",
    "html/css": "html_css",  "html css": "html_css",  "html": "html_css",  "css": "html_css",
    "jest": "jest",  "graphql": "graphql",
    # Web — Backend
    "node.js": "nodejs",  "nodejs": "nodejs",  "node js": "nodejs",
    "flask": "flask",
    "spring boot": "spring_boot",  "springboot": "spring_boot",
    "rest api": "rest_api",  "rest": "rest_api",  "restapi": "rest_api",
    "microservices": "microservices",
    # Databases
    "sql": "sql",
    "mysql": "mysql",  "mysq": "mysql",
    "postgresql": "postgresql",  "postgres": "postgresql",
    "mongodb": "mongodb",  "redis": "redis",
    # DevOps / Cloud
    "docker": "docker",
    "kubernetes": "kubernetes",  "kubernates": "kubernetes",  "k8s": "kubernetes",
    "ci/cd": "ci_cd",  "cicd": "ci_cd",  "ci cd": "ci_cd",
    "aws": "aws",
    # Mobile
    "android": "android",  "firebase": "firebase",
    # CS Fundamentals
    "algorithms": "algorithms",  "algoritms": "algorithms",
    "data structure": "data_structures",  "data structures": "data_structures",
    "competitive programming": "competitive_programming",
    # Design
    "ui/ux": "ui_ux",  "ui ux": "ui_ux",  "figma": "figma",
}

# Pre-sort keys: multi-word / hyphenated phrases FIRST (longest first),
# so they are matched before single-token keys during normalization.
MULTI_WORD_KEYS = sorted(
    [k for k in SKILL_ALIASES if " " in k or "-" in k],
    key=lambda x: -len(x)
)

# ──────────────────────────────────────────────────────────────────────────────
# RAW RESUME DATASET — 10 Candidates
# ──────────────────────────────────────────────────────────────────────────────
RESUMES = [
    ("Arjun Sharma",    "Pyhton, MachineLearning, SQL, pandas, numpy, Deep-learning"),
    ("Priya Nair",      "JavaScrpit, Reacts, Node.JS, MongoDb, REST api, HTML/CSS"),
    ("Rahul Gupta",     "Java, Spring Boot, MySql, Microservices, Docker, kubernates"),
    ("Sneha Patel",     "Python, TensorFlow, Keras, NLP, BERT, data-viz, matplotlib"),
    ("Vikram Singh",    "C++, Algoritms, Data Structure, competitive programming, python"),
    ("Ananya Krishnan", "javascript, vue.js, python, flask, PostgreSQL, AWS, CI/CD"),
    ("Karan Mehta",     "Python, Sklearn, XGboost, feature engineering, SQL, tableau"),
    ("Deepika Rao",     "Java, Android, Kotlin, Firebase, REST, UI/UX, figma"),
    ("Aditya Kumar",    "Reactjs, TypeScrpit, GraphQL, redux, tailwind, nodejs, jest"),
    ("Meera Iyer",      "python, R, statistics, ML, regression, clustering, Power-BI"),
]

# ──────────────────────────────────────────────────────────────────────────────
# JOB DESCRIPTION DATASET — 3 JDs
# ──────────────────────────────────────────────────────────────────────────────
JDS = [
    ("JD-1", "Kakao",  "ML Engineer",
     "Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, SQL, Data Visualization",
     "NLP, BERT, Feature Engineering, Statistics"),
    ("JD-2", "Naver",  "Backend Engineer",
     "Java, Spring Boot, MySQL, PostgreSQL, Microservices, Docker, Kubernetes",
     "REST API, CI/CD, Redis"),
    ("JD-3", "Line",   "Frontend Engineer",
     "JavaScript, React, Vue, TypeScript, REST API, HTML/CSS",
     "Node.js, GraphQL, Redux, Jest, AWS"),
]


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 & 2 — Normalize Skills and Deduplicate
# ──────────────────────────────────────────────────────────────────────────────
def normalize_skills(raw_str):
    """
    1. Split on commas → lowercase tokens
    2. Match multi-word phrases first (longest first)
    3. Apply SKILL_ALIASES; discard unmapped tokens
    4. Remove duplicates (preserve first-seen order)
    """
    tokens = [t.strip().lower() for t in raw_str.split(",")]
    canonical = []
    for token in tokens:
        matched = False
        for key in MULTI_WORD_KEYS:          # multi-word / hyphenated first
            if token == key:
                canonical.append(SKILL_ALIASES[key])
                matched = True
                break
        if not matched and token in SKILL_ALIASES:
            canonical.append(SKILL_ALIASES[token])
        # tokens not in the map are silently discarded

    # Deduplicate — keep first occurrence
    seen, deduped = set(), []
    for s in canonical:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — Build Shared Vocabulary (alphabetically sorted)
# ──────────────────────────────────────────────────────────────────────────────
normalized_resumes = [(name, normalize_skills(raw)) for name, raw in RESUMES]
all_skills = set(skill for _, skills in normalized_resumes for skill in skills)
VOCAB = sorted(all_skills)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — Compute TF-IDF Vectors for Resumes
#   TF(skill, resume) = 1 / N      (each skill appears once after dedup)
#   IDF(skill)        = ln(10 / df(skill))
#   TF-IDF            = TF * IDF
# ──────────────────────────────────────────────────────────────────────────────
# Document frequency
df = {skill: sum(1 for _, skills in normalized_resumes if skill in skills)
      for skill in VOCAB}

# IDF — natural log, no smoothing
idf = {skill: math.log(10 / df[skill]) for skill in VOCAB}

# TF-IDF vectors
def make_tfidf_vector(skills):
    N = len(skills)
    return [((1.0 / N) * idf[skill]) if skill in skills else 0.0
            for skill in VOCAB]

resume_vectors = [(name, make_tfidf_vector(skills))
                  for name, skills in normalized_resumes]


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 — Build JD Binary Vectors over the same VOCAB
# ──────────────────────────────────────────────────────────────────────────────
def make_jd_vector(required_str, preferred_str):
    jd_skills = set(normalize_skills(required_str + ", " + preferred_str))
    return [1 if skill in jd_skills else 0 for skill in VOCAB], jd_skills

jd_vectors = []
for jd_id, company, role, req, pref in JDS:
    vec, skills = make_jd_vector(req, pref)
    jd_vectors.append((jd_id, company, role, vec, skills))


# ──────────────────────────────────────────────────────────────────────────────
# STEP 6 — Cosine Similarity and Top-3 Ranking
#   Cosine(A, B) = (A · B) / (|A| × |B|)
# ──────────────────────────────────────────────────────────────────────────────
def cosine_similarity(a, b):
    dot    = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# OUTPUT
# ──────────────────────────────────────────────────────────────────────────────
for jd_id, company, role, jd_vec, _ in jd_vectors:
    scores = [(name, cosine_similarity(res_vec, jd_vec))
              for name, res_vec in resume_vectors]
    # Sort: descending score; alphabetical by name on tie
    scores.sort(key=lambda x: (-round(x[1], 10), x[0]))
    top3 = scores[:3]

    result_str = ", ".join(f"{name}({score:.2f})" for name, score in top3)

    print(f"{jd_id} Result: Top 3 Candidates with Matching Scores.")
    print(f"{result_str}")
    print()
    print("  Full Ranking (All Candidates):")
    for rank, (name, score) in enumerate(scores, 1):
        marker = " ← TOP 3" if rank <= 3 else ""
        print(f"  {rank:>2}. {name:<20s}  Score: {score:.4f}{marker}")
    print()

print("\n" + "=" * 60)
print("  INTERMEDIATE DIAGNOSTICS")
print("=" * 60)

print("\n[Step 1-2] Normalized + Deduplicated Resume Skills:")
for name, skills in normalized_resumes:
    print(f"  {name:20s} → {skills}")

print(f"\n[Step 3] Vocabulary ({len(VOCAB)} unique skills, alphabetical):")
print(f"  {VOCAB}")

print("\n[Step 4] IDF values:")
for skill in VOCAB:
    print(f"  {skill:30s}  df={df[skill]}  IDF={idf[skill]:.4f}")

print("\n[Step 5] JD Normalized Skill Sets:")
for jd_id, company, role, _, jd_skills in jd_vectors:
    print(f"  {jd_id} ({company}): {sorted(jd_skills)}")