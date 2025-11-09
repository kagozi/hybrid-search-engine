# indexing/index_pg.py
import json
import psycopg2
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np

# Load data
docs = []
with open("../data/arxiv_clean.jsonl") as f:
    for line in f:
        docs.append(json.loads(line))

print(f"Loaded {len(docs)} docs")

# DB
conn = psycopg2.connect(host="localhost", dbname="ir_db", user="user", password="pass")
cur = conn.cursor()

cur.execute("""
DROP TABLE IF EXISTS documents;
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    url TEXT,
    title TEXT,
    text TEXT,
    clean_text TSVECTOR,
    embedding VECTOR(384)
);
""")
conn.commit()

# Embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
batch = 32
embs = []

print("Embedding...")
for i in tqdm(range(0, len(docs), batch)):
    batch_txt = [d["clean_text"] for d in docs[i:i+batch]]
    embs.extend(model.encode(batch_txt, normalize_embeddings=True).tolist())

# Insert
print("Inserting...")
for doc, emb in tqdm(zip(docs, embs), total=len(docs)):
    cur.execute("""
        INSERT INTO documents (url, title, text, clean_text, embedding)
        VALUES (%s, %s, %s, to_tsvector(%s), %s)
    """, (doc["url"], doc["title"], doc["text"], doc["clean_text"], emb))

conn.commit()

# HNSW index
print("Creating HNSW index...")
cur.execute("CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=100);")
conn.commit()
cur.close()
conn.close()
print("Indexing complete!")