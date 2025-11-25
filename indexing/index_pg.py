import json
import psycopg2
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os

# Use environment variables with fallbacks for flexibility
conn = psycopg2.connect(
    host=os.getenv("PGHOST", "db"),  # "db" when in Docker, "localhost" when running locally
    port=int(os.getenv("PGPORT", "5432")),  # 5432 inside Docker network
    dbname=os.getenv("PGDATABASE", "ir_db"),
    user=os.getenv("PGUSER", "postgres"),
    password=os.getenv("PGPASSWORD", "mysecretpassword")
)
cur = conn.cursor()

print("Enabling pgvector and creating table...")
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

cur.execute("""
DROP TABLE IF EXISTS documents;
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
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
        # Use the document's ID if available, otherwise use URL as ID
        doc_id = d.get("id", d.get("url", str(len(docs))))
        emb = model.encode(d["clean_text"], normalize_embeddings=True).tolist()
        docs.append((doc_id, d["title"], d["url"], d["text"], d["clean_text"], emb))

print(f"Inserting {len(docs)} documents...")
cur.executemany("""
INSERT INTO documents (id, title, url, text, clean_text, embedding)
VALUES (%s, %s, %s, %s, to_tsvector(%s), %s)
ON CONFLICT (id) DO UPDATE SET
    title = EXCLUDED.title,
    url = EXCLUDED.url,
    text = EXCLUDED.text,
    clean_text = EXCLUDED.clean_text,
    embedding = EXCLUDED.embedding
""", docs)
conn.commit()

print("Indexing complete! Your hybrid search engine is ready.")
cur.close()
conn.close()