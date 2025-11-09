# api/main.py
from fastapi import FastAPI, Query
from pydantic import BaseModel
import psycopg2
from sentence_transformers import SentenceTransformer, CrossEncoder
from fusion.adaptive_fusion import get_alpha
import numpy as np
from typing import List

app = FastAPI(title="Hybrid Semantic Search + Rerank")

embedder = SentenceTransformer('all-MiniLM-L6-v2')
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

conn = psycopg2.connect(host="db", dbname="ir_db", user="user", password="pass")

class Hit(BaseModel):
    id: int
    title: str
    url: str
    score: float
    alpha: float
    rerank_score: float

def bm25(query: str, limit=100):
    cur = conn.cursor()
    cur.execute("""
        SELECT id, title, url, text, ts_rank_cd(clean_text, plainto_tsquery(%s)) AS s
        FROM documents
        WHERE clean_text @@ plainto_tsquery(%s)
        ORDER BY s DESC LIMIT %s;
    """, (query, query, limit))
    rows = cur.fetchall()
    cur.close()
    return rows

def dense(query: str, limit=100):
    qv = embedder.encode(query, normalize_embeddings=True).tolist()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, title, url, text, embedding <=> %s AS d
        FROM documents
        ORDER BY d LIMIT %s;
    """, (qv, limit))
    rows = cur.fetchall()
    cur.close()
    return rows

def fuse_rerank(query: str, bm25_res, dense_res, top_k=10):
    alpha = get_alpha(query)

    # Normalize
    bm25_s = np.array([r[4] for r in bm25_res]) if bm25_res else np.array([])
    dense_s = np.array([1-r[4] for r in dense_res]) if dense_res else np.array([])
    if bm25_s.size: bm25_s = bm25_s / (bm25_s.max() + 1e-8)
    if dense_s.size: dense_s = dense_s / (dense_s.max() + 1e-8)

    fused = {}
    for i, r in enumerate(bm25_res):
        fused[r[0]] = alpha * bm25_s[i]
    for i, r in enumerate(dense_res):
        fused[r[0]] = fused.get(r[0], 0) + (1-alpha) * dense_s[i]

    candidates = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k*3]

    # Pull full docs for reranking
    pairs = []
    doc_map = {}
    for doc_id, _ in candidates:
        cur = conn.cursor()
        cur.execute("SELECT title, url, text FROM documents WHERE id=%s", (doc_id,))
        title, url, txt = cur.fetchone()
        cur.close()
        pairs.append((query, txt[:1000]))
        doc_map[len(pairs)-1] = (doc_id, title, url)

    rerank_scores = reranker.predict(pairs)
    ranked = sorted(zip(doc_map.items(), rerank_scores), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for (idx, (doc_id, title, url)), score in ranked:
        results.append(Hit(
            id=doc_id, title=title, url=url,
            score=round(score, 4), alpha=round(alpha, 3),
            rerank_score=round(score, 4)
        ))
    return results, alpha

@app.get("/search", response_model=List[Hit])
def search(q: str = Query(..., min_length=1)):
    b = bm25(q)
    d = dense(q)
    hits, _ = fuse_rerank(q, b, d)
    return hits