#!/usr/bin/env python
"""
Train a small regression model to predict the BM25–Dense fusion weight alpha
from query features.

We:

  - Use BEIR datasets (SciFact, TREC-COVID) that are already indexed in Postgres.
  - For each query:
      * Run BM25 and dense retrieval.
      * For a grid of alpha values, fuse BM25 + dense scores.
      * Compute AP@K of the fused ranking vs qrels.
      * Select the alpha* that maximizes AP@K.
  - Extract query features via fusion/query_analyzer.py.
  - Fit a linear regression on logit(alpha*):

        logit(alpha*) ≈ W · x(q) + b

    where x(q) = [length_norm, entity_ratio, acronym_ratio, idf_variance].

  - Print WEIGHTS and BIAS so you can paste them into fusion/adaptive_fusion.py.
"""

import os
import sys
import json
import zipfile
import math
from typing import Dict, List, Tuple, Set

import numpy as np
import psycopg2
import requests
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from query_analyzer import extract_features


# ================================
# CONFIGURATION
# ================================
TRAIN_DATASETS = ["scifact", "trec-covid"]  # BEIR datasets to train alpha on

TOP_K = 100        # retrieval depth for BM25 and dense
AP_AT_K = 100      # cutoff for AP@K
ALPHA_GRID = np.linspace(0.05, 0.95, 19)  # candidate alphas (avoid exact 0/1)

# PostgreSQL connection (same as your eval script)
PG_CONFIG = dict(
    host="localhost",
    port=5433,
    dbname="ir_db",
    user="postgres",
    password="mysecretpassword",
)


# ================================
# DB & Models
# ================================
def get_table_name(dataset_name: str) -> str:
    return f"beir_{dataset_name.replace('-', '_')}"


def connect_db():
    conn = psycopg2.connect(**PG_CONFIG)
    conn.autocommit = True
    return conn, conn.cursor()


# -------------------------
# BEIR dataset loading
# -------------------------
def download_beir_dataset(dataset_name: str) -> str:
    """Download BEIR dataset if not already present; return dataset path."""
    dataset_path = f"data/beir/{dataset_name}"
    if os.path.exists(dataset_path):
        return dataset_path

    print(f"Downloading {dataset_name}...")
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, timeout=120, headers=headers, stream=True)
    r.raise_for_status()

    os.makedirs("data/beir", exist_ok=True)
    zip_path = f"data/beir/{dataset_name}.zip"

    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

    print("Extracting...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall("data/beir")

    os.remove(zip_path)
    return dataset_path


def load_beir_queries_qrels(dataset_name: str):
    """Load queries and qrels (test split) in a robust way."""
    path = download_beir_dataset(dataset_name)

    # queries
    queries_path = f"{path}/queries.jsonl"
    queries: Dict[str, str] = {}
    with open(queries_path) as f:
        for line in f:
            q = json.loads(line)
            qid = str(q["_id"])
            queries[qid] = q["text"]

    # qrels
    qrels_path = f"{path}/qrels/test.tsv"
    qrels_dict: Dict[str, Dict[str, int]] = {}
    with open(qrels_path) as f:
        for i, line in enumerate(f):
            if i == 0 and any(h in line.lower() for h in ["query-id", "qid", "topic"]):
                # header
                continue

            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue

            if len(parts) == 3:
                # BEIR style: qid, docid, rel
                qid, docid, rel = parts
            else:
                # TREC style: qid, Q0, docid, rel, ...
                qid, _q0, docid, rel = parts[:4]

            qid = str(qid)
            docid = str(docid)
            rel = int(rel)

            qrels_dict.setdefault(qid, {})[docid] = rel

    return queries, qrels_dict


# -------------------------
# Retrieval primitives
# -------------------------
from sentence_transformers import SentenceTransformer  # noqa: E402

dense_model = SentenceTransformer("all-MiniLM-L6-v2")


def bm25_search(cur, query: str, table_name: str, limit: int = TOP_K):
    """BM25-style search over TSVECTOR clean_text."""
    cur.execute(f"""
        SELECT id, ts_rank_cd(clean_text, plainto_tsquery(%s)) AS score
        FROM {table_name}
        WHERE clean_text @@ plainto_tsquery(%s)
        ORDER BY score DESC
        LIMIT %s;
    """, (query, query, limit))
    return [(row[0], float(row[1])) for row in cur.fetchall()]


def dense_search(cur, query: str, table_name: str, limit: int = TOP_K):
    """Dense retrieval using cosine distance over PGVECTOR embedding."""
    q_emb = dense_model.encode(query, normalize_embeddings=True)
    cur.execute(f"""
        SELECT id, embedding <=> %s::vector AS dist
        FROM {table_name}
        ORDER BY dist ASC
        LIMIT %s;
    """, (q_emb.tolist(), limit))
    rows = cur.fetchall()
    if not rows:
        return []

    scores = [1.0 - float(r[1]) for r in rows]
    max_score = max(scores) if scores else 1.0
    if max_score <= 0:
        norm_scores = [0.0 for _ in scores]
    else:
        norm_scores = [s / max_score for s in scores]

    return [(rows[i][0], norm_scores[i]) for i in range(len(rows))]


def normalize_scores(results: List[Tuple[str, float]]) -> Dict[str, float]:
    if not results:
        return {}
    max_score = max(score for _, score in results)
    if max_score <= 0:
        return {str(doc_id): 0.0 for doc_id, _ in results}
    return {str(doc_id): score / max_score for doc_id, score in results}


# -------------------------
# Metric: AP@K
# -------------------------
def average_precision_at_k(
    ranked_list: List[str],
    relevant_docs: Set[str],
    k: int,
) -> float:
    """Compute AP@K for a single query."""
    if not relevant_docs:
        return 0.0

    ranked = ranked_list[:k]
    num_rel = 0
    sum_prec = 0.0

    for i, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant_docs:
            num_rel += 1
            sum_prec += num_rel / float(i)

    # divide by number of relevant docs (standard AP definition)
    return sum_prec / float(len(relevant_docs))


# ================================
# Main training loop
# ================================
def main():
    conn, cur = connect_db()

    all_feature_vecs: List[np.ndarray] = []
    all_alpha_stars: List[float] = []

    for dataset_name in TRAIN_DATASETS:
        print("\n" + "=" * 60)
        print(f"Training alpha on {dataset_name.upper()}")
        print("=" * 60)

        table_name = get_table_name(dataset_name)
        # sanity: table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name = %s
            );
        """, (table_name,))
        if not cur.fetchone()[0]:
            print(f"❌ Table {table_name} does not exist. Skipping {dataset_name}.")
            continue

        queries, qrels = load_beir_queries_qrels(dataset_name)

        # restrict to queries with qrels
        qids = [qid for qid in queries.keys() if qid in qrels]

        for qid in tqdm(qids, desc=f"Queries ({dataset_name})"):
            query_text = queries[qid]
            qrel_docs = {doc_id for doc_id, rel in qrels[qid].items() if rel > 0}
            if not qrel_docs:
                # no relevant docs -> no signal
                continue

            # Run retrieval
            try:
                bm25_results = bm25_search(cur, query_text, table_name, TOP_K)
                dense_results = dense_search(cur, query_text, table_name, TOP_K)
            except Exception as e:
                print(f"\nError retrieving for qid={qid}: {e}")
                continue

            if not bm25_results and not dense_results:
                # no candidates -> cannot learn alpha
                continue

            bm25_norm = normalize_scores(bm25_results)
            dense_norm = normalize_scores(dense_results)
            all_docs = set(bm25_norm) | set(dense_norm)
            if not all_docs:
                continue

            # If union of candidates contains no relevant docs, skip this query
            if not (all_docs & qrel_docs):
                continue

            # For each alpha in ALPHA_GRID, compute AP@K, pick best
            best_alpha = None
            best_ap = -1.0

            # Precompute doc list for speed
            doc_ids = list(all_docs)

            for alpha in ALPHA_GRID:
                fused_scores = {}
                for doc_id in doc_ids:
                    s_b = bm25_norm.get(doc_id, 0.0)
                    s_d = dense_norm.get(doc_id, 0.0)
                    fused_scores[doc_id] = alpha * s_b + (1.0 - alpha) * s_d

                ranked = [
                    doc_id
                    for doc_id, _ in sorted(
                        fused_scores.items(), key=lambda x: x[1], reverse=True
                    )
                ]
                ap = average_precision_at_k(ranked, qrel_docs, AP_AT_K)

                if ap > best_ap:
                    best_ap = ap
                    best_alpha = float(alpha)

            # Some queries may still fail to produce a meaningful best_alpha
            if best_alpha is None or best_ap <= 0.0:
                continue

            # Extract features
            f = extract_features(query_text)
            length_norm = f["length"] / 10.0
            x = np.array(
                [
                    length_norm,
                    f["entity_ratio"],
                    f["acronym_ratio"],
                    f["idf_variance"],
                ],
                dtype=float,
            )

            all_feature_vecs.append(x)
            all_alpha_stars.append(best_alpha)

    conn.close()

    if not all_feature_vecs:
        print("No training data collected for alpha. Exiting.")
        return

    X = np.stack(all_feature_vecs, axis=0)
    y_alpha = np.array(all_alpha_stars, dtype=float)

    # Clip alphas away from 0/1 to avoid infinite logit
    eps = 1e-3
    y_alpha_clipped = np.clip(y_alpha, eps, 1.0 - eps)
    y_logit = np.log(y_alpha_clipped / (1.0 - y_alpha_clipped))

    # Fit linear regression: y_logit ≈ W · x + b
    reg = LinearRegression()
    reg.fit(X, y_logit)

    W = reg.coef_
    b = float(reg.intercept_)

    print("\n" + "=" * 60)
    print("Training complete.")
    print("=" * 60)
    print(f"Number of training queries used: {len(y_alpha)}")
    print(f"Alpha* stats: min={y_alpha.min():.3f}, max={y_alpha.max():.3f}, mean={y_alpha.mean():.3f}")
    print("\nPaste these into fusion/adaptive_fusion.py:\n")

    print("WEIGHTS = np.array([")
    for w in W:
        print(f"    {w:.6f},")
    print("], dtype=float)")
    print(f"BIAS = {b:.6f}")

    # Optionally, save to JSON for later inspection
    params = {
        "weights": W.tolist(),
        "bias": b,
        "num_queries": int(len(y_alpha)),
        "alpha_min": float(y_alpha.min()),
        "alpha_max": float(y_alpha.max()),
        "alpha_mean": float(y_alpha.mean()),
    }
    os.makedirs("fusion", exist_ok=True)
    with open("fusion/alpha_params.json", "w") as f:
        json.dump(params, f, indent=2)
    print("\nSaved parameters to fusion/alpha_params.json")


if __name__ == "__main__":
    main()
