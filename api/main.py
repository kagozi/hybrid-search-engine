# api/main.py
from fastapi import FastAPI, Query
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import RealDictCursor
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from typing import List
from fusion.adaptive_fusion import get_alpha
import time

app = FastAPI(title="Hybrid Semantic Search + Rerank")

# Models
embedder = SentenceTransformer('all-MiniLM-L6-v2')
rerank_tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
rerank_model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
rerank_model.eval()

def get_db():
    for _ in range(10):
        try:
            conn = psycopg2.connect(
                host="db",
                dbname="ir_db",
                user="user",
                password="pass",
                cursor_factory=RealDictCursor
            )
            conn.autocommit = True
            return conn
        except psycopg2.OperationalError:
            print("DB not ready, retrying...")
            time.sleep(2)
    raise Exception("Could not connect to DB")

conn = get_db()

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
        SELECT id, title, url, text,
               embedding <=> %s::vector(384) AS d
        FROM documents
        ORDER BY d
        LIMIT %s;
    """, (qv, limit))
    rows = cur.fetchall()
    cur.close()
    return rows


def fuse_rerank(query: str, bm25_res, dense_res, top_k=10):
    alpha = get_alpha(query)

    bm25_s = np.array([r['s'] for r in bm25_res]) if bm25_res else np.array([])
    dense_s = np.array([1-r['d'] for r in dense_res]) if dense_res else np.array([])
    if bm25_s.size: bm25_s = bm25_s / (bm25_s.max() + 1e-8)
    if dense_s.size: dense_s = dense_s / (dense_s.max() + 1e-8)

    fused = {}
    for i, r in enumerate(bm25_res):
        fused[r['id']] = alpha * bm25_s[i]
    for i, r in enumerate(dense_res):
        fused[r['id']] = fused.get(r['id'], 0) + (1-alpha) * dense_s[i]

    candidates = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k*3]

    doc_map = {}
    passages = []
    for doc_id, _ in candidates:
        cur = conn.cursor()
        cur.execute("SELECT title, url, text FROM documents WHERE id=%s", (doc_id,))
        row = cur.fetchone()
        cur.close()
        if row:
            idx = len(passages)
            passages.append(row['text'][:1000])
            doc_map[idx] = (doc_id, row['title'], row['url'])

    if not passages:
        return []

    inputs = rerank_tokenizer(
        [query] * len(passages),
        passages,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    with torch.no_grad():
        logits = rerank_model(**inputs).logits.squeeze(-1)
    scores = torch.sigmoid(logits).cpu().numpy()

    ranked = sorted(zip(doc_map.items(), scores), key=lambda x: x[1], reverse=True)[:top_k]

    return [
        Hit(
            id=doc_id, title=title, url=url,
            score=round(float(score), 4),
            alpha=round(alpha, 3),
            rerank_score=round(float(score), 4)
        )
        for (idx, (doc_id, title, url)), score in ranked
    ]

@app.get("/search", response_model=List[Hit])
def search(q: str = Query(..., min_length=1)):
    b = bm25(q)
    d = dense(q)
    return fuse_rerank(q, b, d)

@app.get("/health")
def health():
    return {"status": "ok", "docs": conn.cursor().execute("SELECT COUNT(*) FROM documents").fetchone()[0]}