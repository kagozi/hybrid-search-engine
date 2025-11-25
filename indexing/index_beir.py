"""
Index BEIR datasets into PostgreSQL for evaluation.

This script:
  - Downloads the BEIR datasets (SciFact, TREC-COVID)
  - Cleans text into a TSVECTOR column for BM25-style search
  - Computes dense embeddings (MiniLM) for vector search
"""

import json
import os
import zipfile
import requests
import psycopg2
from bs4 import BeautifulSoup
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import spacy

# -----------------------------------
# SpaCy for text cleaning
# -----------------------------------
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except Exception:
    print("Downloading spaCy model...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def clean_text(raw: str) -> str:
    """HTML strip + lowercase + lemmatize + stopword/punct removal."""
    if not raw:
        return ""
    soup = BeautifulSoup(raw, "html.parser")
    text = soup.get_text()
    doc = nlp(text.lower())
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
    ]
    return " ".join(tokens)


# -----------------------------------
# Configuration
# -----------------------------------
DATASETS = ["scifact", "trec-covid"]
MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# PostgreSQL connection
conn = psycopg2.connect(
    host="localhost",
    port=5433,   # adjust if needed
    dbname="ir_db",
    user="postgres",
    password="mysecretpassword",
)
conn.autocommit = True
cur = conn.cursor()


# -----------------------------------
# Helpers
# -----------------------------------
def download_beir_dataset(dataset_name: str) -> str:
    """Download BEIR dataset if not already present; return dataset path."""
    dataset_path = f"data/beir/{dataset_name}"
    if os.path.exists(dataset_path):
        print(f"Dataset {dataset_name} already exists -> {dataset_path}")
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


def create_dataset_table(dataset_name: str) -> str:
    """Create a separate table for each BEIR dataset."""
    table_name = f"beir_{dataset_name.replace('-', '_')}"
    print(f"Creating table {table_name}...")

    cur.execute(f"""
        DROP TABLE IF EXISTS {table_name};
        CREATE TABLE {table_name} (
            id TEXT PRIMARY KEY,
            title TEXT,
            text TEXT,
            clean_text TSVECTOR,
            embedding VECTOR(384)
        );
    """)

    # GIN index for BM25/TS search
    cur.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{table_name}_clean_text
        ON {table_name} USING GIN (clean_text);
    """)

    # HNSW index for dense retrieval
    cur.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{table_name}_embedding
        ON {table_name} USING hnsw (embedding vector_cosine_ops);
    """)

    return table_name


def index_beir_dataset(dataset_name: str) -> None:
    """Index a BEIR dataset into PostgreSQL."""
    print("\n" + "=" * 60)
    print(f"Indexing {dataset_name.upper()}")
    print("=" * 60)

    # 1) Download
    dataset_path = download_beir_dataset(dataset_name)

    # 2) Create table
    table_name = create_dataset_table(dataset_name)

    # 3) Load corpus
    corpus_path = f"{dataset_path}/corpus.jsonl"
    if not os.path.exists(corpus_path):
        print(f"WARNING: corpus.jsonl not found at {corpus_path}")
        return

    print(f"Loading corpus from {corpus_path}...")
    with open(corpus_path) as f:
        first_doc = json.loads(f.readline())
        print("Sample document keys:", list(first_doc.keys()))
        print("Sample _id:", first_doc.get("_id"))

    docs = []
    doc_ids_seen = set()

    with open(corpus_path) as f:
        for line in tqdm(f, desc=f"Processing {dataset_name} docs"):
            doc = json.loads(line)
            doc_id = str(doc["_id"])

            if doc_id in doc_ids_seen:
                # BEIR should not have duplicates; but just in case
                continue
            doc_ids_seen.add(doc_id)

            title = doc.get("title", "")
            text = doc.get("text", "")
            full_text = f"{title} {text}".strip()

            # cleaned for BM25-style search
            cleaned = clean_text(full_text)
            if not cleaned:
                cleaned = full_text.lower()

            # dense embedding on original text
            embedding = MODEL.encode(full_text, normalize_embeddings=True).tolist()

            docs.append((doc_id, title, text, cleaned, embedding))

    print(f"Processed {len(docs)} unique documents")
    if docs:
        print("Sample IDs:", [docs[i][0] for i in range(min(5, len(docs)))])

    print(f"Inserting {len(docs)} documents into {table_name}...")
    cur.executemany(f"""
        INSERT INTO {table_name} (id, title, text, clean_text, embedding)
        VALUES (%s, %s, %s, to_tsvector(%s), %s)
    """, docs)

    print(f"✓ Successfully indexed {len(docs)} documents into {table_name}")

    # quick verification
    cur.execute(f"SELECT id FROM {table_name} LIMIT 5")
    db_ids = [row[0] for row in cur.fetchall()]
    print("Sample IDs from DB:", db_ids)


# -----------------------------------
# Main
# -----------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("BEIR Dataset Indexing Script")
    print("=" * 60)
    print(f"Datasets to index: {', '.join(DATASETS)}\n")

    for ds in DATASETS:
        try:
            index_beir_dataset(ds)
        except Exception as e:
            print(f"Error indexing {ds}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("INDEXING COMPLETE!")
    print("=" * 60)

    # table summary
    print("\nDatabase tables (public schema):")
    cur.execute("""
        SELECT table_name,
               pg_size_pretty(pg_total_relation_size(quote_ident(table_name))) AS size
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_type = 'BASE TABLE'
        ORDER BY table_name;
    """)
    for row in cur.fetchall():
        print(f"  - {row[0]}: {row[1]}")

    cur.close()
    conn.close()
