#!/usr/bin/env python
"""
Improved alpha training with better regularization and dataset-specific models.

Key improvements:
1. Per-dataset alpha models (different characteristics for SciFact vs TREC-COVID vs NFCorpus)
2. Ridge regression to prevent overfitting
3. More robust feature engineering
4. Validation split to tune hyperparameters
5. Support for 3 datasets
"""

import os
import json
import math
import zipfile
from typing import Dict, List, Tuple

import numpy as np
import psycopg2
import requests
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

# Import query_analyzer from parent directory
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fusion.query_analyzer import extract_features


# ================================
# CONFIGURATION
# ================================
TRAIN_DATASETS = ["scifact", "trec-covid", "nfcorpus"]
TOP_K = 200
AP_AT_K = 100
ALPHA_GRID = np.linspace(0.05, 0.95, 19)

PG_CONFIG = dict(
    host="localhost",
    port=5433,
    dbname="ir_db",
    user="postgres",
    password="mysecretpassword",
)

dense_model = SentenceTransformer("all-MiniLM-L6-v2")


def get_table_name(dataset_name: str) -> str:
    return f"beir_{dataset_name.replace('-', '_')}"


def connect_db():
    conn = psycopg2.connect(**PG_CONFIG)
    conn.autocommit = True
    return conn, conn.cursor()


def download_beir_dataset(dataset_name: str) -> str:
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
    
    with zipfile.ZipFile(zip_path) as z:
        z.extractall("data/beir")
    
    os.remove(zip_path)
    return dataset_path


def load_beir_queries_qrels(dataset_name: str):
    path = download_beir_dataset(dataset_name)
    
    queries_path = f"{path}/queries.jsonl"
    queries = {}
    with open(queries_path) as f:
        for line in f:
            q = json.loads(line)
            queries[str(q["_id"])] = q["text"]
    
    qrels_path = f"{path}/qrels/test.tsv"
    qrels_dict = {}
    with open(qrels_path) as f:
        for i, line in enumerate(f):
            if i == 0 and any(h in line.lower() for h in ["query-id", "qid"]):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            
            if len(parts) == 3:
                qid, docid, rel = parts
            else:
                qid, _q0, docid, rel = parts[:4]
            
            qrels_dict.setdefault(str(qid), {})[str(docid)] = int(rel)
    
    return queries, qrels_dict


def bm25_search(cur, query: str, table_name: str, limit: int):
    """Improved BM25 with title weighting."""
    cur.execute(f"""
        SELECT id, 
               ts_rank_cd(
                   setweight(to_tsvector('english', COALESCE(title, '')), 'A') ||
                   setweight(to_tsvector('english', COALESCE(text, '')), 'B'),
                   plainto_tsquery('english', %s),
                   32
               ) AS score
        FROM {table_name}
        WHERE (
            setweight(to_tsvector('english', COALESCE(title, '')), 'A') ||
            setweight(to_tsvector('english', COALESCE(text, '')), 'B')
        ) @@ plainto_tsquery('english', %s)
        ORDER BY score DESC LIMIT %s;
    """, (query, query, limit))
    return [(row[0], float(row[1])) for row in cur.fetchall()]


def dense_search(cur, query: str, table_name: str, limit: int):
    q_emb = dense_model.encode(query, normalize_embeddings=True)
    cur.execute(f"""
        SELECT id, 1 - (embedding <=> %s::vector) AS similarity
        FROM {table_name}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (q_emb.tolist(), q_emb.tolist(), limit))
    return [(row[0], float(row[1])) for row in cur.fetchall()]


def normalize_scores(results):
    """Min-max normalization."""
    if not results:
        return {}
    scores = [score for _, score in results]
    min_s, max_s = min(scores), max(scores)
    if max_s - min_s < 1e-9:
        return {str(doc_id): 1.0 for doc_id, _ in results}
    return {str(doc_id): (score - min_s) / (max_s - min_s) for doc_id, score in results}


def average_precision_at_k(ranked_list, relevant_docs, k):
    if not relevant_docs:
        return 0.0
    
    ranked = ranked_list[:k]
    num_rel = 0
    sum_prec = 0.0
    
    for i, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant_docs:
            num_rel += 1
            sum_prec += num_rel / float(i)
    
    return sum_prec / float(len(relevant_docs))


def extract_enhanced_features(query: str, dataset_name: str):
    """Extract features with dataset-specific indicators."""
    base_features = extract_features(query)
    
    # Dataset-specific dummy features (one-hot encoding)
    is_scifact = 1.0 if dataset_name == "scifact" else 0.0
    is_trec = 1.0 if dataset_name == "trec-covid" else 0.0
    is_nfcorpus = 1.0 if dataset_name == "nfcorpus" else 0.0
    
    return np.array([
        base_features["length"] / 10.0,
        base_features["content_length"] / 10.0,
        base_features["entity_ratio"],
        base_features["acronym_ratio"],
        base_features["idf_variance"] / 10.0,
        is_scifact,
        is_trec,
        is_nfcorpus,
    ], dtype=float)


def main():
    conn, cur = connect_db()
    
    # Collect training data per dataset
    dataset_data = {}
    
    for dataset_name in TRAIN_DATASETS:
        print("\n" + "=" * 60)
        print(f"Collecting training data from {dataset_name.upper()}")
        print("=" * 60)
        
        table_name = get_table_name(dataset_name)
        
        # Check if table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = %s
            );
        """, (table_name,))
        
        if not cur.fetchone()[0]:
            print(f"❌ Table {table_name} does not exist. Skipping.")
            print(f"   Please run: python indexing/index_beir_3datasets.py")
            continue
        
        # Verify table has documents
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        doc_count = cur.fetchone()[0]
        print(f"✓ Table {table_name} has {doc_count:,} documents")
        
        queries, qrels = load_beir_queries_qrels(dataset_name)
        qids = [qid for qid in queries.keys() if qid in qrels]
        
        print(f"Processing {len(qids)} queries...")
        
        feature_vecs = []
        alpha_stars = []
        
        for qid in tqdm(qids, desc=f"Training on {dataset_name}"):
            query_text = queries[qid]
            qrel_docs = {doc_id for doc_id, rel in qrels[qid].items() if rel > 0}
            
            if not qrel_docs:
                continue
            
            try:
                bm25_results = bm25_search(cur, query_text, table_name, TOP_K)
                dense_results = dense_search(cur, query_text, table_name, TOP_K)
            except Exception as e:
                # print(f"\nError retrieving for qid={qid}: {e}")
                continue
            
            if not bm25_results and not dense_results:
                continue
            
            bm25_norm = normalize_scores(bm25_results)
            dense_norm = normalize_scores(dense_results)
            all_docs = set(bm25_norm) | set(dense_norm)
            
            if not all_docs or not (all_docs & qrel_docs):
                continue
            
            # Find best alpha via grid search
            best_alpha = None
            best_ap = -1.0
            doc_ids = list(all_docs)
            
            for alpha in ALPHA_GRID:
                fused = {}
                for doc_id in doc_ids:
                    s_b = bm25_norm.get(doc_id, 0.0)
                    s_d = dense_norm.get(doc_id, 0.0)
                    fused[doc_id] = alpha * s_b + (1 - alpha) * s_d
                
                ranked = [d for d, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)]
                ap = average_precision_at_k(ranked, qrel_docs, AP_AT_K)
                
                if ap > best_ap:
                    best_ap = ap
                    best_alpha = float(alpha)
            
            if best_alpha is None or best_ap <= 0.0:
                continue
            
            x = extract_enhanced_features(query_text, dataset_name)
            feature_vecs.append(x)
            alpha_stars.append(best_alpha)
        
        dataset_data[dataset_name] = {
            "features": np.array(feature_vecs),
            "alphas": np.array(alpha_stars)
        }
        print(f"✓ Collected {len(alpha_stars)} training examples for {dataset_name}")
        if len(alpha_stars) > 0:
            print(f"  Alpha range: [{np.min(alpha_stars):.3f}, {np.max(alpha_stars):.3f}]")
            print(f"  Alpha mean: {np.mean(alpha_stars):.3f}")
    
    conn.close()
    
    # Combine all data
    all_features = []
    all_alphas = []
    
    for data in dataset_data.values():
        if len(data["features"]) > 0:
            all_features.append(data["features"])
            all_alphas.append(data["alphas"])
    
    if not all_features:
        print("\n❌ No training data collected!")
        print("Please ensure you have:")
        print("  1. Indexed all datasets (run index_beir_3datasets.py)")
        print("  2. PostgreSQL is running on port 5433")
        print("  3. Database 'ir_db' exists")
        return
    
    X = np.vstack(all_features)
    y_alpha = np.concatenate(all_alphas)
    
    print("\n" + "=" * 60)
    print("COMBINED TRAINING DATA")
    print("=" * 60)
    print(f"Total samples: {len(y_alpha)}")
    print(f"Alpha range: [{y_alpha.min():.3f}, {y_alpha.max():.3f}]")
    print(f"Alpha mean: {y_alpha.mean():.3f}")
    print(f"Alpha std: {y_alpha.std():.3f}")
    
    # Clip and logit transform
    eps = 1e-3
    y_alpha_clipped = np.clip(y_alpha, eps, 1.0 - eps)
    y_logit = np.log(y_alpha_clipped / (1.0 - y_alpha_clipped))
    
    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_logit, test_size=0.2, random_state=42
    )
    
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING (Ridge Regression)")
    print("=" * 60)
    
    # Try different regularization strengths
    best_alpha_param = None
    best_val_mse = float('inf')
    
    for alpha_reg in [0.01, 0.1, 1.0, 10.0, 100.0]:
        reg = Ridge(alpha=alpha_reg)
        reg.fit(X_train, y_train)
        y_val_pred = reg.predict(X_val)
        val_mse = np.mean((y_val - y_val_pred) ** 2)
        print(f"Ridge alpha={alpha_reg:>6.2f}: Validation MSE={val_mse:.4f}")
        
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_alpha_param = alpha_reg
    
    print(f"\n✓ Best regularization: alpha={best_alpha_param}")
    
    # Final model on all data with best hyperparameter
    final_model = Ridge(alpha=best_alpha_param)
    final_model.fit(X, y_logit)
    
    W = final_model.coef_
    b = float(final_model.intercept_)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total samples: {len(y_alpha)}")
    print(f"Feature dimension: {len(W)}")
    print(f"Validation MSE: {best_val_mse:.4f}")
    
    print("\nFeature Weights:")
    print("-" * 60)
    feature_names = [
        "length_norm", "content_length_norm", "entity_ratio",
        "acronym_ratio", "idf_variance_norm", "is_scifact", "is_trec_covid", "is_nfcorpus"
    ]
    for i, (name, weight) in enumerate(zip(feature_names, W)):
        direction = "→ BM25" if weight > 0 else "→ Dense"
        print(f"  {name:<25} {weight:>8.4f}  {direction}")
    print(f"  {'bias':<25} {b:>8.4f}")
    
    # Interpret dataset effects
    print("\nDataset-Specific Effects:")
    print("-" * 60)
    if W[5] > 0:
        print(f"  SciFact:     +{W[5]:.4f} → Favors BM25")
    else:
        print(f"  SciFact:     {W[5]:.4f} → Favors Dense")
    
    if W[6] > 0:
        print(f"  TREC-COVID:  +{W[6]:.4f} → Favors BM25")
    else:
        print(f"  TREC-COVID:  {W[6]:.4f} → Favors Dense")
    
    if W[7] > 0:
        print(f"  NFCorpus:    +{W[7]:.4f} → Favors BM25")
    else:
        print(f"  NFCorpus:    {W[7]:.4f} → Favors Dense")
    
    print("\n" + "=" * 60)
    print("COPY THESE VALUES TO fusion/adaptive_fusion.py:")
    print("=" * 60)
    print("\nWEIGHTS = np.array([")
    for w in W:
        print(f"    {w:.6f},")
    print("], dtype=float)")
    print(f"\nBIAS = {b:.6f}")
    
    # Save parameters
    params = {
        "weights": W.tolist(),
        "bias": b,
        "feature_names": feature_names,
        "num_samples": int(len(y_alpha)),
        "alpha_stats": {
            "min": float(y_alpha.min()),
            "max": float(y_alpha.max()),
            "mean": float(y_alpha.mean()),
            "std": float(y_alpha.std()),
        },
        "validation_mse": float(best_val_mse),
        "regularization": best_alpha_param,
        "datasets": {
            ds: {
                "samples": int(len(dataset_data[ds]["alphas"])),
                "alpha_mean": float(np.mean(dataset_data[ds]["alphas"])),
            }
            for ds in TRAIN_DATASETS if ds in dataset_data and len(dataset_data[ds]["alphas"]) > 0
        }
    }
    
    os.makedirs("fusion", exist_ok=True)
    with open("fusion/alpha_params_improved.json", "w") as f:
        json.dump(params, f, indent=2)
    
    print("\n✅ Saved parameters to fusion/alpha_params_improved.json")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()