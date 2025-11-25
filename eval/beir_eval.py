# # eval/beir_eval.py
# import json
# import psycopg2
# import requests
# import zipfile
# import os
# from sentence_transformers import SentenceTransformer, CrossEncoder
# from tqdm import tqdm
# import numpy as np
# from ranx import Qrels, Run, evaluate
# import sys
# import pathlib

# # Add project root to path
# sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
# from fusion.adaptive_fusion import get_alpha

# # ================================
# # CONFIGURATION
# # ================================
# DATASETS = ["scifact", "trec-covid"]
# TOP_K = 100
# RERANK_K = 30
# FINAL_K = 10

# # Connect to PostgreSQL
# conn = psycopg2.connect(
#     host="localhost",
#     port=5433,
#     dbname="ir_db",
#     user="postgres",
#     password="mysecretpassword"
# )
# conn.autocommit = True
# cur = conn.cursor()

# # Models
# dense_model = SentenceTransformer('all-MiniLM-L6-v2')
# reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# # Map dataset names to table names
# def get_table_name(dataset_name):
#     return f"beir_{dataset_name.replace('-', '_')}"

# def download_beir_dataset(dataset_name):
#     """Download BEIR dataset metadata (queries and qrels only)"""
#     dataset_path = f"data/beir/{dataset_name}"
#     if os.path.exists(dataset_path):
#         print(f"Dataset {dataset_name} already exists, skipping download.")
#         return dataset_path
    
#     print(f"Downloading {dataset_name}...")
#     url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    
#     headers = {'User-Agent': 'Mozilla/5.0'}
#     r = requests.get(url, timeout=120, headers=headers, stream=True)
#     r.raise_for_status()
    
#     os.makedirs("data/beir", exist_ok=True)
#     zip_path = f"data/beir/{dataset_name}.zip"
    
#     with open(zip_path, 'wb') as f:
#         for chunk in r.iter_content(chunk_size=8192):
#             f.write(chunk)
    
#     print(f"Extracting...")
#     with zipfile.ZipFile(zip_path) as z:
#         z.extractall("data/beir")
    
#     os.remove(zip_path)
#     return dataset_path

# def bm25_search(query, table_name, limit=TOP_K):
#     """BM25 search on specified table"""
#     cur.execute(f"""
#         SELECT id, ts_rank_cd(clean_text, plainto_tsquery(%s)) AS score
#         FROM {table_name}
#         WHERE clean_text @@ plainto_tsquery(%s)
#         ORDER BY score DESC LIMIT %s
#     """, (query, query, limit))
#     return [(row[0], float(row[1])) for row in cur.fetchall()]

# def dense_search(query, table_name, limit=TOP_K):
#     """Dense search on specified table"""
#     q_emb = dense_model.encode(query, normalize_embeddings=True)
#     cur.execute(f"""
#         SELECT id, embedding <=> %s::vector AS dist
#         FROM {table_name}
#         ORDER BY dist LIMIT %s
#     """, (q_emb.tolist(), limit))
#     results = cur.fetchall()
    
#     if not results:
#         return []
    
#     scores = [1 - row[1] for row in results]
#     max_score = max(scores) if scores else 1
#     normalized = [s / max_score for s in scores]
#     return [(row[0], normalized[i]) for i, row in enumerate(results)]

# def get_document_text(doc_id, table_name):
#     """Retrieve document text for reranking"""
#     cur.execute(f"SELECT text FROM {table_name} WHERE id = %s", (doc_id,))
#     row = cur.fetchone()
#     return row[0] if row else ""

# def rerank_documents(query, doc_ids, table_name):
#     """Rerank documents using cross-encoder"""
#     passages = [get_document_text(doc_id, table_name)[:1000] for doc_id in doc_ids]
#     pairs = [[query, passage] for passage in passages if passage]
    
#     if not pairs:
#         return []
    
#     # Get raw logits from cross-encoder
#     scores = reranker.predict(pairs)
    
#     # Convert logits to positive scores using sigmoid
#     # This ensures scores are between 0 and 1
#     import math
#     positive_scores = [1 / (1 + math.exp(-score)) for score in scores]
    
#     return positive_scores

# def load_dataset(dataset_name):
#     """Load queries and qrels"""
#     path = download_beir_dataset(dataset_name)
    
#     queries_path = f"{path}/queries.jsonl"
#     with open(queries_path) as f:
#         # CRITICAL: Keep query IDs as strings exactly as they appear
#         queries = {}
#         for line in f:
#             q = json.loads(line)
#             qid = str(q['_id'])  # Ensure string
#             queries[qid] = q['text']
    
#     print(f"Loaded {len(queries)} queries. Sample query IDs: {list(queries.keys())[:5]}")
    
#     qrels_path = f"{path}/qrels/test.tsv"
#     with open(qrels_path) as f:
#         qrels_dict = {}
#         for i, line in enumerate(f):
#             # Skip header if present
#             if i == 0 and any(h in line.lower() for h in ["query-id", "qid", "topic"]):
#                 continue

#             parts = line.strip().split('\t')
#             if len(parts) < 3:
#                 continue

#             if len(parts) == 3:
#                 # BEIR-style: query-id corpus-id score
#                 qid, docid, rel = parts
#             else:
#                 # TREC-style: query-id Q0 docid rel [iteration...]
#                 qid, _q0, docid, rel = parts[:4]

#             qid = str(qid)
#             docid = str(docid)
#             rel = int(rel)

#             qrels_dict.setdefault(qid, {})[docid] = rel

    
#     print(f"Loaded {len(qrels_dict)} qrels. Sample qrel query IDs: {list(qrels_dict.keys())[:5]}")
    
#     return queries, qrels_dict

# def rrf_fusion(bm25_results, dense_results, k=60):
#     """Reciprocal Rank Fusion"""
#     scores = {}
#     for i, (doc_id, _) in enumerate(bm25_results):
#         scores[str(doc_id)] = scores.get(str(doc_id), 0) + 1 / (i + k)
#     for i, (doc_id, _) in enumerate(dense_results):
#         scores[str(doc_id)] = scores.get(str(doc_id), 0) + 1 / (i + k)
#     return scores

# # ================================
# # EVALUATION LOOP
# # ================================
# results_summary = {}

# for dataset_name in DATASETS:
#     print(f"\n{'='*60}")
#     print(f"=== Evaluating on {dataset_name.upper()} ===")
#     print(f"{'='*60}")
    
#     table_name = get_table_name(dataset_name)
    
#     # Check if table exists
#     cur.execute("""
#         SELECT EXISTS (
#             SELECT FROM information_schema.tables 
#             WHERE table_schema = 'public' 
#             AND table_name = %s
#         );
#     """, (table_name,))
    
#     if not cur.fetchone()[0]:
#         print(f"❌ Table {table_name} does not exist!")
#         print(f"Please run: python indexing/index_beir.py")
#         continue
    
#     # Verify table has documents
#     cur.execute(f"SELECT COUNT(*) FROM {table_name}")
#     doc_count = cur.fetchone()[0]
#     print(f"✓ Using table: {table_name} ({doc_count:,} documents)")
    
#     try:
#         queries, qrels_dict = load_dataset(dataset_name)
#         qrels = Qrels(qrels_dict)
#         some_qid = next(iter(qrels_dict.keys()))
#         print("Sample qid:", some_qid)
#         print("Sample qrel docids for that qid:", list(qrels_dict[some_qid].keys())[:10])

#         table_name = get_table_name(dataset_name)
#         for did in list(qrels_dict[some_qid].keys())[:10]:
#             cur.execute(f"SELECT COUNT(*) FROM {table_name} WHERE id = %s", (did,))
#             count = cur.fetchone()[0]
#             print(did, "exists in DB?" , count > 0)

#         run_bm25 = {}
#         run_dense = {}
#         run_rrf = {}
#         run_adaptive = {}
        
#         for qid, query in tqdm(queries.items(), desc="Running queries"):
#             try:
#                 # 1. BM25
#                 bm25_results = bm25_search(query, table_name, TOP_K)
#                 run_bm25[qid] = {str(doc_id): score for doc_id, score in bm25_results} if bm25_results else {}
                
#                 # 2. Dense
#                 dense_results = dense_search(query, table_name, TOP_K)
#                 run_dense[qid] = {str(doc_id): score for doc_id, score in dense_results} if dense_results else {}
                
#                 # 3. RRF
#                 rrf_scores = rrf_fusion(bm25_results, dense_results)
#                 run_rrf[qid] = rrf_scores if rrf_scores else {}
                
#                 # 4. Adaptive Fusion + Re-rank
#                 alpha = get_alpha(query)
#                 fused = {}
#                 bm25_dict = {doc_id: score for doc_id, score in bm25_results}
#                 dense_dict = {doc_id: score for doc_id, score in dense_results}
#                 all_ids = set(bm25_dict) | set(dense_dict)
                
#                 for doc_id in all_ids:
#                     s_bm25 = bm25_dict.get(doc_id, 0)
#                     s_dense = dense_dict.get(doc_id, 0)
#                     fused[doc_id] = alpha * s_bm25 + (1 - alpha) * s_dense
                
#                 candidates = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:RERANK_K]
#                 candidate_ids = [doc_id for doc_id, _ in candidates]
                
#                 if candidate_ids:
#                     rerank_scores = rerank_documents(query, candidate_ids, table_name)
#                     final_scored = sorted(zip(candidate_ids, rerank_scores), key=lambda x: x[1], reverse=True)[:FINAL_K]
#                     run_adaptive[qid] = {str(doc_id): score for doc_id, score in final_scored}
#                 else:
#                     run_adaptive[qid] = {}
                    
#             except Exception as e:
#                 print(f"\nError processing query {qid}: {e}")
#                 continue
        
#         # Save runs
#         os.makedirs("eval/output", exist_ok=True)
#         Run(run_bm25).save(f"eval/output/{dataset_name}_bm25.json")
#         Run(run_dense).save(f"eval/output/{dataset_name}_dense.json")
#         Run(run_rrf).save(f"eval/output/{dataset_name}_rrf.json")
#         Run(run_adaptive).save(f"eval/output/{dataset_name}_adaptive.json")
        
#         # Evaluate
#         print(f"\nEvaluating {dataset_name.upper()}...")
#         metrics = ["ndcg@10", "recall@100", "map@100"]
#         eval_results = {}
        
#         bm25_run = Run(run_bm25)
#         bm25_scores = evaluate(qrels, bm25_run, metrics, make_comparable=True)
#         eval_results["BM25"] = bm25_scores
        
#         dense_run = Run(run_dense)
#         dense_scores = evaluate(qrels, dense_run, metrics, make_comparable=True)
#         eval_results["Dense (MiniLM)"] = dense_scores
        
#         rrf_run = Run(run_rrf)
#         rrf_scores = evaluate(qrels, rrf_run, metrics, make_comparable=True)
#         eval_results["RRF"] = rrf_scores
        
#         adaptive_run = Run(run_adaptive)
#         adaptive_scores = evaluate(qrels, adaptive_run, metrics, make_comparable=True)
#         eval_results["Adaptive Fusion + Re-rank (Yours)"] = adaptive_scores
        
#         results_summary[dataset_name] = eval_results
        
#         print(f"\nResults on {dataset_name.upper()}:")
#         print("-" * 60)
#         for method, scores in eval_results.items():
#             print(f"{method}:")
#             for metric, value in scores.items():
#                 print(f"  {metric}: {value:.4f}")
#         print("-" * 60)
        
#     except Exception as e:
#         print(f"Error processing {dataset_name}: {e}")
#         import traceback
#         traceback.print_exc()
#         continue

# # Summary Table
# print("\n" + "="*60)
# print("=== SUMMARY TABLE FOR IEEE REPORT ===")
# print("="*60)
# print("| Method                        | SciFact nDCG@10 | TREC-COVID nDCG@10 |")
# print("|-------------------------------|-----------------|--------------------|")
# for method in ["BM25", "Dense (MiniLM)", "RRF", "Adaptive Fusion + Re-rank (Yours)"]:
#     sci = results_summary.get("scifact", {}).get(method, {}).get("ndcg@10", "N/A")
#     trec = results_summary.get("trec-covid", {}).get(method, {}).get("ndcg@10", "N/A")
#     if isinstance(sci, float) and isinstance(trec, float):
#         print(f"| {method:<29} | {sci:>15.4f} | {trec:>18.4f} |")
#     else:
#         print(f"| {method:<29} | {str(sci):>15} | {str(trec):>18} |")

# print("\nEvaluation complete! Results saved in eval/output/")

# cur.close()
# conn.close()

# eval/beir_eval.py

"""
Evaluate hybrid retrieval on BEIR datasets (SciFact, TREC-COVID) using:

  Retrieval:
    - BM25 (Postgres tsvector)
    - Dense (MiniLM)
    - RRF hybrid (BM25 + dense)
    - Adaptive alpha fusion (BM25 + dense, learned alpha)

  Re-ranking:
    - CrossEncoder over top-K RRF candidates (RRF + CE)
"""

import json
import os
import zipfile
import sys
import pathlib

import psycopg2
import requests
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder
from ranx import Qrels, Run, evaluate

# Ensure we can import fusion.*
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fusion.adaptive_fusion import get_alpha

# ================================
# CONFIGURATION
# ================================
DATASETS = ["scifact", "trec-covid"]
TOP_K = 100         # initial retrieval depth per method
RERANK_K = 30       # number of candidates passed to cross-encoder
FINAL_K = 10        # depth for final ranking (nDCG@10 etc.)

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
    """Download BEIR dataset (queries + qrels + corpus) if not present."""
    dataset_path = f"data/beir/{dataset_name}"
    if os.path.exists(dataset_path):
        print(f"Dataset {dataset_name} already exists, skipping download.")
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


# -------------------------
# Retrieval primitives
# -------------------------
def bm25_search(query: str, table_name: str, limit: int = TOP_K):
    """BM25-style search with ts_rank_cd over TSVECTOR clean_text."""
    cur.execute(f"""
        SELECT id, ts_rank_cd(clean_text, plainto_tsquery(%s)) AS score
        FROM {table_name}
        WHERE clean_text @@ plainto_tsquery(%s)
        ORDER BY score DESC
        LIMIT %s;
    """, (query, query, limit))
    return [(row[0], float(row[1])) for row in cur.fetchall()]


def dense_search(query: str, table_name: str, limit: int = TOP_K):
    """Dense retrieval with cosine distance over PGVECTOR embedding."""
    q_emb = dense_model.encode(query, normalize_embeddings=True)
    cur.execute(f"""
        SELECT id, embedding <=> %s::vector AS dist
        FROM {table_name}
        ORDER BY dist ASC
        LIMIT %s;
    """, (q_emb.tolist(), limit))
    results = cur.fetchall()
    if not results:
        return []

    # convert cosine distance to similarity
    scores = [1.0 - float(row[1]) for row in results]
    max_score = max(scores) if scores else 1.0
    if max_score <= 0:
        normalized = [0.0 for _ in scores]
    else:
        normalized = [s / max_score for s in scores]

    return [(row[0], normalized[i]) for i, row in enumerate(results)]


def get_document_text(doc_id: str, table_name: str) -> str:
    """Retrieve document text; truncate later in reranker."""
    cur.execute(f"SELECT text FROM {table_name} WHERE id = %s", (doc_id,))
    row = cur.fetchone()
    return row[0] if row and row[0] else ""


def rerank_documents(query: str, doc_ids, table_name: str):
    """Re-rank given doc_ids using cross-encoder.

    Returns list of scores aligned with doc_ids.
    """
    passages = [get_document_text(doc_id, table_name)[:1000] for doc_id in doc_ids]
    pairs = [[query, passage] for passage in passages if passage]

    if not pairs:
        return []

    scores = reranker.predict(pairs)
    # Optionally squash to [0, 1] via sigmoid
    import math
    positive_scores = [1.0 / (1.0 + math.exp(-s)) for s in scores]
    return positive_scores


def normalize_scores(results):
    """Normalize a list of (doc_id, score) to [0, 1]."""
    if not results:
        return {}
    max_score = max(score for _, score in results)
    if max_score <= 0:
        return {doc_id: 0.0 for doc_id, _ in results}
    return {doc_id: score / max_score for doc_id, score in results}


def rrf_fusion(bm25_results, dense_results, k: int = 60):
    """Reciprocal Rank Fusion of two ranked lists."""
    scores = {}
    for i, (doc_id, _) in enumerate(bm25_results):
        scores.setdefault(str(doc_id), 0.0)
        scores[str(doc_id)] += 1.0 / (i + k)
    for i, (doc_id, _) in enumerate(dense_results):
        scores.setdefault(str(doc_id), 0.0)
        scores[str(doc_id)] += 1.0 / (i + k)
    return scores


# -------------------------
# Dataset loading
# -------------------------
def load_dataset(dataset_name: str):
    """Load queries and qrels in a robust way (3- or 4-column qrels)."""
    path = download_beir_dataset(dataset_name)

    # queries
    queries_path = f"{path}/queries.jsonl"
    queries = {}
    with open(queries_path) as f:
        for line in f:
            q = json.loads(line)
            qid = str(q["_id"])
            queries[qid] = q["text"]
    print(f"Loaded {len(queries)} queries. Sample query IDs: {list(queries.keys())[:5]}")

    # qrels
    qrels_path = f"{path}/qrels/test.tsv"
    qrels_dict = {}
    with open(qrels_path) as f:
        for i, line in enumerate(f):
            # skip header if present
            if i == 0 and any(h in line.lower() for h in ["query-id", "qid", "topic"]):
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

    print(f"Loaded {len(qrels_dict)} qrels. Sample qrel query IDs: {list(qrels_dict.keys())[:5]}")
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

    # check table exists
    cur.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name = %s
        );
    """, (table_name,))
    if not cur.fetchone()[0]:
        print(f"❌ Table {table_name} does not exist!")
        print("Please run the indexing script first.")
        continue

    # table size
    cur.execute(f"SELECT COUNT(*) FROM {table_name}")
    doc_count = cur.fetchone()[0]
    print(f"✓ Using table: {table_name} ({doc_count:,} documents)")

    try:
        queries, qrels_dict = load_dataset(dataset_name)
        qrels = Qrels(qrels_dict)

        run_bm25 = {}
        run_dense = {}
        run_rrf = {}
        run_adaptive_retrieval = {}      # alpha-fused BM25 + dense
        run_rrf_ce = {}                  # RRF + CrossEncoder (our main method)

        # quick sanity: docid in DB?
        sample_qid = next(iter(qrels_dict.keys()))
        sample_docid = next(iter(qrels_dict[sample_qid].keys()))
        print(f"Sanity check qid: {sample_qid}")
        print(f"Sample qrel docid for that qid: {sample_docid}")
        cur.execute(f"SELECT COUNT(*) FROM {table_name} WHERE id = %s", (sample_docid,))
        print("Exists in DB?", cur.fetchone()[0] > 0)

        for qid, query in tqdm(queries.items(), desc="Running queries"):
            try:
                # 1) BM25
                bm25_results = bm25_search(query, table_name, TOP_K)
                run_bm25[qid] = {str(doc_id): score for doc_id, score in bm25_results} if bm25_results else {}

                # 2) Dense
                dense_results = dense_search(query, table_name, TOP_K)
                run_dense[qid] = {str(doc_id): score for doc_id, score in dense_results} if dense_results else {}

                # 3) RRF
                rrf_scores = rrf_fusion(bm25_results, dense_results)
                run_rrf[qid] = rrf_scores if rrf_scores else {}

                # 4) Adaptive alpha fusion (BM25 + dense)
                bm25_norm = normalize_scores(bm25_results)
                dense_norm = normalize_scores(dense_results)
                all_ids = set(bm25_norm) | set(dense_norm)

                alpha = get_alpha(query, dataset_name=dataset_name)
                fused_scores = {}
                for doc_id in all_ids:
                    s_bm25 = bm25_norm.get(doc_id, 0.0)
                    s_dense = dense_norm.get(doc_id, 0.0)
                    fused_scores[str(doc_id)] = alpha * s_bm25 + (1.0 - alpha) * s_dense
                run_adaptive_retrieval[qid] = fused_scores

                # 5) RRF + CrossEncoder re-ranking (candidate pool from RRF)
                if rrf_scores:
                    # sort by RRF score and take top RERANK_K
                    sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
                    candidates = sorted_rrf[:RERANK_K]
                    candidate_ids = [doc_id for doc_id, _ in candidates]
                else:
                    # fallback: use adaptive fused scores if RRF empty
                    sorted_fused = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
                    candidate_ids = [doc_id for doc_id, _ in sorted_fused[:RERANK_K]]

                if candidate_ids:
                    ce_scores = rerank_documents(query, candidate_ids, table_name)
                    # align lengths just in case some documents had empty passages
                    final_pairs = list(zip(candidate_ids, ce_scores))[:FINAL_K]
                    run_rrf_ce[qid] = {str(doc_id): float(score) for doc_id, score in final_pairs}
                else:
                    run_rrf_ce[qid] = {}

            except Exception as e:
                print(f"\nError processing query {qid}: {e}")
                continue

        # Save runs
        os.makedirs("eval/output", exist_ok=True)
        Run(run_bm25).save(f"eval/output/{dataset_name}_bm25.json")
        Run(run_dense).save(f"eval/output/{dataset_name}_dense.json")
        Run(run_rrf).save(f"eval/output/{dataset_name}_rrf.json")
        Run(run_adaptive_retrieval).save(f"eval/output/{dataset_name}_adaptive.json")
        Run(run_rrf_ce).save(f"eval/output/{dataset_name}_rrf_ce.json")

        # Evaluate
        print(f"\nEvaluating {dataset_name.upper()}...")
        metrics = ["ndcg@10", "recall@100", "map@100"]
        eval_results = {}

        bm25_run = Run(run_bm25)
        eval_results["BM25"] = evaluate(qrels, bm25_run, metrics, make_comparable=True)

        dense_run = Run(run_dense)
        eval_results["Dense (MiniLM)"] = evaluate(qrels, dense_run, metrics, make_comparable=True)

        rrf_run = Run(run_rrf)
        eval_results["RRF (BM25 + Dense)"] = evaluate(qrels, rrf_run, metrics, make_comparable=True)

        adaptive_run = Run(run_adaptive_retrieval)
        eval_results["Adaptive Fusion (alpha)"] = evaluate(qrels, adaptive_run, metrics, make_comparable=True)

        rrf_ce_run = Run(run_rrf_ce)
        eval_results["RRF + CrossEncoder (Ours)"] = evaluate(qrels, rrf_ce_run, metrics, make_comparable=True)

        results_summary[dataset_name] = eval_results

        # Pretty-print
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
        continue


# -------------------------
# Summary table for paper
# -------------------------
print("\n" + "=" * 60)
print("=== SUMMARY TABLE FOR IEEE REPORT (nDCG@10) ===")
print("=" * 60)
print("| Method                      | SciFact nDCG@10 | TREC-COVID nDCG@10 |")
print("|-----------------------------|-----------------|--------------------|")

method_order = [
    "BM25",
    "Dense (MiniLM)",
    "RRF (BM25 + Dense)",
    "Adaptive Fusion (alpha)",
    "RRF + CrossEncoder (Ours)",
]

for method in method_order:
    sci = results_summary.get("scifact", {}).get(method, {}).get("ndcg@10", "N/A")
    trec = results_summary.get("trec-covid", {}).get(method, {}).get("ndcg@10", "N/A")

    if isinstance(sci, float) and isinstance(trec, float):
        print(f"| {method:<27} | {sci:>15.4f} | {trec:>18.4f} |")
    else:
        print(f"| {method:<27} | {str(sci):>15} | {str(trec):>18} |")

print("\nEvaluation complete! Results saved in eval/output/")

cur.close()
conn.close()
