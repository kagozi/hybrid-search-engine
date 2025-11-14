# Hybrid Semantic Search Engine with Adaptive Fusion & Neural Re-ranking  
---

## Project Overview  

This project implements a **lightweight, high-performance hybrid information retrieval (IR) system** that combines:  

- **Classical sparse retrieval** (BM25 via PostgreSQL `tsvector`)  
- **Dense semantic retrieval** (`all-MiniLM-L6-v2` embeddings + `pgvector`)  
- **Neural re-ranking** (`cross-encoder/ms-marco-MiniLM-L-6-v2`)  
- **Novel contribution**: **Adaptive Fusion** — dynamically blends sparse and dense scores based on query complexity  

> **Domain**: Scientific literature (arXiv CS + Quantitative Biology)  
> **Dataset**: 10,000+ arXiv papers (crawled or pre-loaded)  
> **Evaluation**: Ready for BEIR (`ranx`) — SciFact, TREC-COVID  

---

## Key Features  

| Feature | Implementation |
|--------|----------------|
| **Hybrid Retrieval** | BM25 + Dense (HNSW) |
| **Adaptive Fusion** | Query-aware `α` via length, entities, acronyms |
| **Neural Re-ranking** | Cross-encoder (top-30 → top-10) |
| **Scalable Indexing** | PostgreSQL + `pgvector` + GIN index |
| **REST API** | FastAPI + OpenAPI (`/docs`) |
| **Dockerized** | Full reproducibility on macOS (Apple Silicon) |

---

## System Architecture  

```
┌─────────────────┐
│     Client      │
│  (curl / UI)    │
└───────┬─────────┘
        │ GET /search?q=...
        ▼
┌─────────────────┐
│    FastAPI      │
│   (api/main.py) │
└───────┬─────────┘
        │
 ┌──────┼──────┐
 │      │      │
 ▼      ▼      ▼
BM25  Dense  Fusion
 │      │      │
 ▼      ▼      ▼
PostgreSQL (pgvector)
        │
        ▼
   Documents + Embeddings
```

## Quick Start (One-Click)  

```bash
# 1. Clone & enter
git clone https://github.com/kagozi/hybrid-search-engine.git
cd hybrid-search-engine

# 2. Crawl 10K arXiv papers
python crawler/crawler.py

# 3. Start DB + Index
docker-compose up -d db && sleep 10
docker-compose up index

# 4. Run API
docker-compose up --build api
```

**API Ready:** [http://localhost:8000/docs](http://localhost:8000/docs)  

Try:  
```text
q=CRISPR off-target prediction
q=quantum error correction
```

---

## Manual Run (No Docker)  

```bash
# Install deps
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Crawl → Clean → Index
python crawler/crawler.py
python indexing/ingest.py
python indexing/index_pg.py

# Run API
uvicorn api.main:app --reload --port 8000
```

---
## Evaluation (BEIR + `ranx`)  
## Novel Contribution: Adaptive Fusion  

```python
α = get_alpha(query)  # 0.0 (dense-heavy) → 1.0 (BM25-heavy)
score = α × bm25_norm + (1−α) × dense_norm
```

- **Short, technical queries** → favor BM25  
- **Long, descriptive queries** → favor dense  
- **Re-ranked** with cross-encoder for final precision  
---

## Technical Stack  

| Layer | Technology |
|------|------------|
| **Backend** | Python 3.11, FastAPI |
| **Database** | PostgreSQL + `pgvector` |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Re-ranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Indexing** | BM25 (`tsvector`), HNSW (`pgvector`) |
| **Deployment** | Docker Compose |

## References  

- BEIR: A Heterogeneous Benchmark for Information Retrieval  
- ColBERT, SPLADE, and Cross-Encoders  
- PostgreSQL `pgvector` Documentation  
- arXiv Dataset (Kuleshov et al.)  