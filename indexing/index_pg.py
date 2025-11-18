# indexing/index_pg.py
import json
import psycopg2
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

# CONNECT TO DOCKER DB VIA SERVICE NAME
conn = psycopg2.connect(
    host="db",  # <-- DOCKER SERVICE NAME
    dbname="ir_db",
    user="user",
    password="pass",
    port=5432
)
cur = conn.cursor()

print("Creating table...")
cur.execute("""
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    title TEXT,
    url TEXT,
    text TEXT,
    clean_text TSVECTOR,
    embedding FLOAT[]
);
CREATE INDEX IF NOT EXISTS idx_clean_text ON documents USING GIN (clean_text);
""")
conn.commit()

model = SentenceTransformer('all-MiniLM-L6-v2')
docs = []
input_path = "data/arxiv_clean.jsonl"  # <-- relative to container

print(f"Loading {input_path}...")
with open(input_path) as f:
    for line in tqdm(f, desc="Indexing", unit="doc"):
        d = json.loads(line)
        embedding = model.encode(d["clean_text"], normalize_embeddings=True).tolist()
        docs.append((
            d["title"], d["url"], d["text"], d["clean_text"], embedding
        ))

print(f"Inserting {len(docs)} documents...")
cur.executemany("""
INSERT INTO documents (title, url, text, clean_text, embedding)
VALUES (%s, %s, %s, to_tsvector(%s), %s)
""", docs)
conn.commit()

print("Indexing complete!")
cur.close()
conn.close()