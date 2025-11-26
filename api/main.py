# api/main.py
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware  # ← NEW: Import CORS
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
import os

app = FastAPI(title="Hybrid Semantic Search + Rerank (arXiv)")

# ← NEW: Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],  # Allow your Vue dev server (and * for simplicity)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Models (load once at startup)
embedder = SentenceTransformer('all-MiniLM-L6-v2')
rerank_tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
rerank_model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
rerank_model.eval()

def get_db():
    """Connect to DB with retries."""
    for _ in range(10):
        try:
            conn = psycopg2.connect(
                host=os.getenv("PGHOST", "db"),
                port=os.getenv("PGPORT", "5432"),
                dbname=os.getenv("PGDATABASE", "ir_db"),
                user=os.getenv("PGUSER", "postgres"),
                password=os.getenv("PGPASSWORD", "mysecretpassword"),
                cursor_factory=RealDictCursor
            )
            conn.autocommit = True
            return conn
        except psycopg2.OperationalError:
            print("Waiting for DB...")
            time.sleep(2)
    raise Exception("DB connection failed after retries")

conn = get_db()

# Pydantic model (id as str for arXiv IDs like "2401.12345")
class Hit(BaseModel):
    id: str  # ← String for arXiv IDs
    title: str
    url: str
    score: float
    alpha: float
    rerank_score: float

def bm25(query: str, limit: int = 100):
    """Sparse retrieval with BM25."""
    cur = conn.cursor()
    cur.execute("""
        SELECT id, title, url, text, ts_rank_cd(clean_text, plainto_tsquery(%s)) AS s
        FROM documents
        WHERE clean_text @@ plainto_tsquery(%s)
        ORDER BY s DESC
        LIMIT %s;
    """, (query, query, limit))
    rows = cur.fetchall()
    cur.close()
    return rows

def dense(query: str, limit: int = 100):
    """Dense retrieval with embeddings."""
    qv = embedder.encode(query, normalize_embeddings=True).tolist()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, title, url, text,
               embedding <=> %s::vector AS d
        FROM documents
        ORDER BY d
        LIMIT %s;
    """, (qv, limit))
    rows = cur.fetchall()
    cur.close()
    return rows

def fuse_rerank(query: str, bm25_res, dense_res, top_k: int = 10):
    """Adaptive fusion + cross-encoder reranking."""
    alpha = get_alpha(query)

    # Normalize scores
    bm25_scores = np.array([r['s'] for r in bm25_res]) if bm25_res else np.array([])
    dense_scores = np.array([1 - r['d'] for r in dense_res]) if dense_res else np.array([])

    if bm25_scores.size > 0:
        bm25_scores /= (bm25_scores.max() + 1e-8)
    if dense_scores.size > 0:
        dense_scores /= (dense_scores.max() + 1e-8)

    # Fuse scores (handle duplicates)
    fused = {}
    for i, r in enumerate(bm25_res):
        fused[r['id']] = fused.get(r['id'], 0) + alpha * bm25_scores[i]
    for i, r in enumerate(dense_res):
        fused[r['id']] = fused.get(r['id'], 0) + (1 - alpha) * dense_scores[i]

    # Top candidates for reranking
    candidates = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k * 3]

    passages = []
    doc_map = {}
    for doc_id, _ in candidates:
        cur = conn.cursor()
        cur.execute("SELECT title, url, text FROM documents WHERE id = %s", (doc_id,))
        row = cur.fetchone()
        cur.close()
        if row:
            idx = len(passages)
            passages.append(row['text'][:1000])  # Truncate for efficiency
            doc_map[idx] = (doc_id, row['title'], row['url'])

    if not passages:
        return []

    # Cross-encoder reranking
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
    scores = torch.sigmoid(logits).cpu().numpy().tolist()

    # Final ranked results
    ranked = sorted(zip(doc_map.items(), scores), key=lambda x: x[1], reverse=True)[:top_k]

    return [
        Hit(
            id=doc_id,
            title=title,
            url=url,
            score=round(float(score), 4),
            alpha=round(alpha, 3),
            rerank_score=round(float(score), 4)
        )
        for (idx, (doc_id, title, url)), score in ranked
    ]

# Endpoints
@app.get("/search", response_model=List[Hit])
def search(q: str = Query(..., min_length=1)):
    bm25_results = bm25(q)
    dense_results = dense(q)
    return fuse_rerank(q, bm25_results, dense_results)

@app.get("/health")
def health():
    try:
        count = conn.cursor().execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        return {"status": "healthy", "documents_indexed": int(count)}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

# Root for quick testing
@app.get("/")
def root():
    return {"message": "Hybrid Semantic Search Engine (arXiv Corpus) – CSC 785", "docs": "/docs"}