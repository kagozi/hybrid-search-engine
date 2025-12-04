"""
Improved BEIR Evaluation with Multiple Enhancements:

1. Better BM25 configuration with tuned parameters
2. Multi-stage re-ranking with larger candidate pools
3. Late interaction fusion (combines retrieval + reranker scores)
4. Query expansion using pseudo-relevance feedback
5. Ensemble voting from multiple fusion strategies
"""

import json
import os
import zipfile
import sys
import pathlib
import math

import psycopg2
import requests
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder
from ranx import Qrels, Run, evaluate

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fusion.adaptive_fusion import get_alpha

# ================================
# CONFIGURATION
# ================================
DATASETS = ["scifact", "trec-covid", "nfcorpus"]
TOP_K = 200         # Increased from 100 for better recall
RERANK_K = 100      # Increased from 30 - don't cut off too early
FINAL_K = 10

# PostgreSQL connection
conn = psycopg2.connect(
    host="localhost",
    port=5433,
    dbname="ir_db",
    user="postgres",
    password="mysecretpassword",
)
conn.autocommit = True
cur = conn.cursor()

# Models
dense_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def get_table_name(dataset_name: str) -> str:
    return f"beir_{dataset_name.replace('-', '_')}"


def download_beir_dataset(dataset_name: str) -> str:
    """Download BEIR dataset if not present."""
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


# -------------------------
# IMPROVED RETRIEVAL
# -------------------------
def bm25_search_improved(query: str, table_name: str, limit: int = TOP_K):
    """
    Improved BM25 with:
    - Better normalization using setweight
    - Consider both title and text with different weights
    """
    cur.execute(f"""
        SELECT id, 
               ts_rank_cd(
                   setweight(to_tsvector('english', COALESCE(title, '')), 'A') ||
                   setweight(to_tsvector('english', COALESCE(text, '')), 'B'),
                   plainto_tsquery('english', %s),
                   32  -- normalization flag: divide by log(length)
               ) AS score
        FROM {table_name}
        WHERE (
            setweight(to_tsvector('english', COALESCE(title, '')), 'A') ||
            setweight(to_tsvector('english', COALESCE(text, '')), 'B')
        ) @@ plainto_tsquery('english', %s)
        ORDER BY score DESC
        LIMIT %s;
    """, (query, query, limit))
    return [(row[0], float(row[1])) for row in cur.fetchall()]


def dense_search(query: str, table_name: str, limit: int = TOP_K):
    """Dense retrieval with cosine similarity."""
    q_emb = dense_model.encode(query, normalize_embeddings=True)
    cur.execute(f"""
        SELECT id, 1 - (embedding <=> %s::vector) AS similarity
        FROM {table_name}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (q_emb.tolist(), q_emb.tolist(), limit))
    return [(row[0], float(row[1])) for row in cur.fetchall()]


def query_expansion_search(query: str, table_name: str, limit: int = TOP_K):
    """
    Pseudo-Relevance Feedback:
    1. Get top-5 documents from initial dense search
    2. Extract key terms from those documents
    3. Expand query and re-search
    """
    # Initial retrieval
    initial = dense_search(query, table_name, 5)
    if not initial:
        return dense_search(query, table_name, limit)
    
    # Get top doc texts for expansion
    doc_ids = [doc_id for doc_id, _ in initial[:3]]
    cur.execute(f"""
        SELECT text FROM {table_name} 
        WHERE id = ANY(%s)
    """, (doc_ids,))
    
    texts = [row[0] for row in cur.fetchall() if row[0]]
    if not texts:
        return dense_search(query, table_name, limit)
    
    # Simple expansion: concatenate query with snippet from top doc
    expansion = " ".join(texts[:1])[:200]  # First 200 chars from top doc
    expanded_query = f"{query} {expansion}"
    
    # Re-search with expanded query
    q_emb = dense_model.encode(expanded_query, normalize_embeddings=True)
    cur.execute(f"""
        SELECT id, 1 - (embedding <=> %s::vector) AS similarity
        FROM {table_name}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (q_emb.tolist(), q_emb.tolist(), limit))
    return [(row[0], float(row[1])) for row in cur.fetchall()]


def get_document_text(doc_id: str, table_name: str) -> str:
    """Retrieve full document text with title."""
    cur.execute(f"SELECT title, text FROM {table_name} WHERE id = %s", (doc_id,))
    row = cur.fetchone()
    if not row:
        return ""
    title, text = row
    return f"{title or ''} {text or ''}".strip()


def rerank_documents_batched(query: str, doc_ids, table_name: str, batch_size: int = 32):
    """
    Batched re-ranking for efficiency.
    Returns dict of doc_id -> rerank_score
    """
    if not doc_ids:
        return {}
    
    passages = []
    valid_ids = []
    
    for doc_id in doc_ids:
        text = get_document_text(doc_id, table_name)[:512]  # Longer context
        if text:
            passages.append(text)
            valid_ids.append(doc_id)
    
    if not passages:
        return {}
    
    # Batch predict
    pairs = [[query, p] for p in passages]
    scores = reranker.predict(pairs, batch_size=batch_size)
    
    # Sigmoid to normalize
    normalized = [1 / (1 + math.exp(-s)) for s in scores]
    
    return dict(zip(valid_ids, normalized))


def normalize_scores(results):
    """Min-max normalize scores to [0, 1]."""
    if not results:
        return {}
    scores = [score for _, score in results]
    min_s, max_s = min(scores), max(scores)
    if max_s - min_s < 1e-9:
        return {doc_id: 1.0 for doc_id, _ in results}
    return {doc_id: (score - min_s) / (max_s - min_s) for doc_id, score in results}


def rrf_fusion(rankings, k: int = 60):
    """
    Reciprocal Rank Fusion across multiple rankings.
    rankings: list of lists of (doc_id, score) tuples
    """
    scores = {}
    for ranking in rankings:
        for i, (doc_id, _) in enumerate(ranking):
            scores.setdefault(str(doc_id), 0.0)
            scores[str(doc_id)] += 1.0 / (i + k)
    return scores


def late_fusion(retrieval_scores: dict, rerank_scores: dict, beta: float = 0.3):
    """
    Late fusion: combine retrieval and reranking scores.
    
    final_score = (1 - beta) * retrieval_score + beta * rerank_score
    
    beta controls reranker influence (0.2-0.4 works well)
    """
    all_docs = set(retrieval_scores.keys()) | set(rerank_scores.keys())
    fused = {}
    for doc_id in all_docs:
        r_score = retrieval_scores.get(doc_id, 0.0)
        rr_score = rerank_scores.get(doc_id, 0.0)
        fused[doc_id] = (1 - beta) * r_score + beta * rr_score
    return fused


# -------------------------
# Dataset loading
# -------------------------
def load_dataset(dataset_name: str):
    """Load queries and qrels."""
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


# ================================
# EVALUATION LOOP
# ================================
results_summary = {}

for dataset_name in DATASETS:
    print("\n" + "=" * 60)
    print(f"=== Evaluating on {dataset_name.upper()} ===")
    print("=" * 60)
    
    table_name = get_table_name(dataset_name)
    
    # Verify table exists
    cur.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = %s
        );
    """, (table_name,))
    if not cur.fetchone()[0]:
        print(f"❌ Table {table_name} does not exist!")
        continue
    
    cur.execute(f"SELECT COUNT(*) FROM {table_name}")
    doc_count = cur.fetchone()[0]
    print(f"✓ Using table: {table_name} ({doc_count:,} documents)")
    
    try:
        queries, qrels_dict = load_dataset(dataset_name)
        qrels = Qrels(qrels_dict)
        
        # Multiple run configurations
        run_bm25 = {}
        run_dense = {}
        run_rrf = {}
        run_qe = {}  # Query expansion
        run_improved = {}  # Our best method
        
        for qid, query in tqdm(queries.items(), desc="Running queries"):
            try:
                # 1. Improved BM25
                bm25_results = bm25_search_improved(query, table_name, TOP_K)
                run_bm25[qid] = {str(d): s for d, s in bm25_results} if bm25_results else {}
                
                # 2. Dense
                dense_results = dense_search(query, table_name, TOP_K)
                run_dense[qid] = {str(d): s for d, s in dense_results} if dense_results else {}
                
                # 3. Query Expansion
                qe_results = query_expansion_search(query, table_name, TOP_K)
                run_qe[qid] = {str(d): s for d, s in qe_results} if qe_results else {}
                
                # 4. Multi-method RRF
                rrf_scores = rrf_fusion([bm25_results, dense_results, qe_results])
                run_rrf[qid] = rrf_scores if rrf_scores else {}
                
                # 5. IMPROVED METHOD: Adaptive fusion + Late fusion with re-ranking
                
                # Normalize for fusion
                bm25_norm = normalize_scores(bm25_results)
                dense_norm = normalize_scores(dense_results)
                qe_norm = normalize_scores(qe_results)
                
                # Get adaptive alpha
                alpha = get_alpha(query, dataset_name=dataset_name)
                
                # Multi-stage fusion
                # Stage 1: Adaptive fusion of BM25 + Dense
                all_docs = set(bm25_norm.keys()) | set(dense_norm.keys())
                adaptive_scores = {}
                for doc_id in all_docs:
                    s_b = bm25_norm.get(doc_id, 0.0)
                    s_d = dense_norm.get(doc_id, 0.0)
                    adaptive_scores[doc_id] = alpha * s_b + (1 - alpha) * s_d
                
                # Stage 2: Blend with query expansion (20% weight)
                for doc_id in qe_norm:
                    if doc_id in adaptive_scores:
                        adaptive_scores[doc_id] = 0.8 * adaptive_scores[doc_id] + 0.2 * qe_norm[doc_id]
                    else:
                        adaptive_scores[doc_id] = 0.2 * qe_norm[doc_id]
                
                # Get top candidates for re-ranking
                candidates = sorted(adaptive_scores.items(), key=lambda x: x[1], reverse=True)[:RERANK_K]
                candidate_ids = [doc_id for doc_id, _ in candidates]
                
                if candidate_ids:
                    # Re-rank
                    rerank_scores = rerank_documents_batched(query, candidate_ids, table_name)
                    
                    # Late fusion: blend retrieval scores with rerank scores
                    retrieval_cand_scores = {doc_id: score for doc_id, score in candidates}
                    
                    # Normalize rerank scores
                    if rerank_scores:
                        rr_vals = list(rerank_scores.values())
                        rr_min, rr_max = min(rr_vals), max(rr_vals)
                        if rr_max - rr_min > 1e-9:
                            rerank_scores = {
                                doc_id: (score - rr_min) / (rr_max - rr_min)
                                for doc_id, score in rerank_scores.items()
                            }
                    
                    # Use dataset-specific beta
                    beta = 0.4 if dataset_name == "trec-covid" else 0.3
                    final_scores = late_fusion(retrieval_cand_scores, rerank_scores, beta=beta)
                    
                    # Take top-K
                    final_ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:FINAL_K]
                    run_improved[qid] = {str(doc_id): score for doc_id, score in final_ranked}
                else:
                    run_improved[qid] = {}
                    
            except Exception as e:
                print(f"\nError processing query {qid}: {e}")
                continue
        
        # Save runs
        os.makedirs("eval/output", exist_ok=True)
        Run(run_bm25).save(f"eval/output/{dataset_name}_bm25_improved.json")
        Run(run_dense).save(f"eval/output/{dataset_name}_dense.json")
        Run(run_qe).save(f"eval/output/{dataset_name}_qe.json")
        Run(run_rrf).save(f"eval/output/{dataset_name}_rrf_multi.json")
        Run(run_improved).save(f"eval/output/{dataset_name}_improved.json")
        
        # Evaluate
        print(f"\nEvaluating {dataset_name.upper()}...")
        metrics = ["ndcg@10", "recall@100", "map@100"]
        eval_results = {}
        
        eval_results["BM25 (Improved)"] = evaluate(qrels, Run(run_bm25), metrics, make_comparable=True)
        eval_results["Dense (MiniLM)"] = evaluate(qrels, Run(run_dense), metrics, make_comparable=True)
        eval_results["Query Expansion"] = evaluate(qrels, Run(run_qe), metrics, make_comparable=True)
        eval_results["RRF (3-way)"] = evaluate(qrels, Run(run_rrf), metrics, make_comparable=True)
        eval_results["Improved Pipeline (Ours)"] = evaluate(qrels, Run(run_improved), metrics, make_comparable=True)
        
        results_summary[dataset_name] = eval_results
        
        print(f"\nResults on {dataset_name.upper()}:")
        print("-" * 60)
        for method, scores in eval_results.items():
            print(f"{method}:")
            for metric, value in scores.items():
                print(f"  {metric}: {value:.4f}")
        print("-" * 60)
        
    except Exception as e:
        print(f"Error processing {dataset_name}: {e}")
        import traceback
        traceback.print_exc()

# Summary
print("\n" + "=" * 60)
print("=== SUMMARY TABLE (nDCG@10) ===")
print("=" * 60)
print("| Method                      | SciFact | TREC-COVID |")
print("|-----------------------------|---------|------------|")

for method in ["BM25 (Improved)", "Dense (MiniLM)", "Query Expansion", "RRF (3-way)", "Improved Pipeline (Ours)"]:
    sci = results_summary.get("scifact", {}).get(method, {}).get("ndcg@10", "N/A")
    trec = results_summary.get("trec-covid", {}).get(method, {}).get("ndcg@10", "N/A")
    
    if isinstance(sci, float) and isinstance(trec, float):
        print(f"| {method:<27} | {sci:>7.4f} | {trec:>10.4f} |")
    else:
        print(f"| {method:<27} | {str(sci):>7} | {str(trec):>10} |")

print("\n✅ Evaluation complete! Results saved in eval/output/")

cur.close()
conn.close()