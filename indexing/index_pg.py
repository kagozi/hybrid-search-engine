# indexing/index_pg.py
import json
import psycopg2
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

conn = psycopg2.connect(host="db", dbname="ir_db", user="user", password="pass")
cur = conn.cursor()

print("Enabling pgvector and creating table...")
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")  # ← THIS IS REQUIRED

cur.execute("""
DROP TABLE IF EXISTS documents;
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    title TEXT,
    url TEXT,
    text TEXT,
    clean_text TSVECTOR,
    embedding VECTOR(384)
);
CREATE INDEX IF NOT EXISTS idx_clean_text ON documents USING GIN (clean_text);
CREATE INDEX IF NOT EXISTS idx_embedding ON documents USING hnsw (embedding vector_cosine_ops);
""")
conn.commit()

model = SentenceTransformer('all-MiniLM-L6-v2')
docs = []

print("Loading data/arxiv_clean.jsonl...")
with open("data/arxiv_clean.jsonl") as f:
    for line in tqdm(f, desc="Indexing", unit="doc"):
        d = json.loads(line)
        emb = model.encode(d["clean_text"], normalize_embeddings=True).tolist()
        docs.append((d["title"], d["url"], d["text"], d["clean_text"], emb))

print(f"Inserting {len(docs)} documents...")
cur.executemany("""
INSERT INTO documents (title, url, text, clean_text, embedding)
VALUES (%s, %s, %s, to_tsvector(%s), %s)
""", docs)
conn.commit()

print("Indexing complete! Your hybrid search engine is ready.")
cur.close()
conn.close()