"""
Index BEIR datasets into PostgreSQL for evaluation
Run this once to prepare the evaluation datasets
"""
import json
import psycopg2
import requests
import zipfile
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from bs4 import BeautifulSoup
import spacy

# Load spaCy for text cleaning
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except:
    print("Downloading spaCy model...")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def clean_text(raw):
    if not raw:
        return ""
    soup = BeautifulSoup(raw, "html.parser")
    text = soup.get_text()
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
    return " ".join(tokens)

# Configuration
DATASETS = ["scifact", "trec-covid"]
MODEL = SentenceTransformer('all-MiniLM-L6-v2')

# Database connection (adjust port for Mac if needed)
conn = psycopg2.connect(
    host="localhost",
    port=5433,  # or 5432 if running inside Docker
    dbname="ir_db",
    user="postgres",
    password="mysecretpassword"
)
conn.autocommit = True
cur = conn.cursor()

def download_beir_dataset(dataset_name):
    """Download BEIR dataset if not already present"""
    dataset_path = f"data/beir/{dataset_name}"
    if os.path.exists(dataset_path):
        print(f"Dataset {dataset_name} already exists")
        return dataset_path
    
    print(f"Downloading {dataset_name}...")
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    r = requests.get(url, timeout=120, headers=headers, stream=True)
    r.raise_for_status()
    
    os.makedirs("data/beir", exist_ok=True)
    zip_path = f"data/beir/{dataset_name}.zip"
    
    with open(zip_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Extracting...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall("data/beir")
    
    os.remove(zip_path)
    return dataset_path

def create_dataset_table(dataset_name):
    """Create a separate table for each BEIR dataset"""
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
    
    # Create indexes
    cur.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{table_name}_clean_text 
        ON {table_name} USING GIN (clean_text);
    """)
    cur.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{table_name}_embedding 
        ON {table_name} USING hnsw (embedding vector_cosine_ops);
    """)
    
    return table_name

def index_beir_dataset(dataset_name):
    """Index a BEIR dataset into PostgreSQL"""
    print(f"\n{'='*60}")
    print(f"Indexing {dataset_name.upper()}")
    print(f"{'='*60}")
    
    # Download dataset
    dataset_path = download_beir_dataset(dataset_name)
    
    # Create table
    table_name = create_dataset_table(dataset_name)
    
    # Load corpus
    corpus_path = f"{dataset_path}/corpus.jsonl"
    if not os.path.exists(corpus_path):
        print(f"Warning: Corpus file not found at {corpus_path}")
        return
    
    print(f"Loading corpus from {corpus_path}...")
    
    # First, check format of first document
    with open(corpus_path) as f:
        first_doc = json.loads(f.readline())
        print(f"Sample document keys: {list(first_doc.keys())}")
        print(f"Sample _id: {first_doc.get('_id')}")
    
    docs = []
    doc_ids_seen = set()
    
    with open(corpus_path) as f:
        for line in tqdm(f, desc="Processing documents"):
            doc = json.loads(line)
            
            # Use _id field (confirmed from corpus check)
            doc_id = str(doc['_id'])
            
            # Debug: Check for duplicate IDs
            if doc_id in doc_ids_seen:
                print(f"Warning: Duplicate ID {doc_id}")
                continue
            doc_ids_seen.add(doc_id)
            
            title = doc.get('title', '')
            text = doc.get('text', '')
            
            # Combine title and text
            full_text = f"{title} {text}".strip()
            
            # Clean text for BM25
            cleaned = clean_text(full_text)
            if not cleaned:
                # Fallback if cleaning removes everything
                cleaned = full_text.lower()
            
            # Generate embedding on original text (better quality)
            embedding = MODEL.encode(full_text, normalize_embeddings=True).tolist()
            
            docs.append((doc_id, title, text, cleaned, embedding))
    
    print(f"Processed {len(docs)} unique documents")
    print(f"Sample IDs: {[docs[i][0] for i in range(min(5, len(docs)))]}")
    
    print(f"Inserting {len(docs)} documents into {table_name}...")
    cur.executemany(f"""
        INSERT INTO {table_name} (id, title, text, clean_text, embedding)
        VALUES (%s, %s, %s, to_tsvector(%s), %s)
    """, docs)
    
    print(f"✓ Successfully indexed {len(docs)} documents into {table_name}")
    
    # Verify with actual IDs
    cur.execute(f"SELECT id FROM {table_name} LIMIT 5")
    db_ids = [row[0] for row in cur.fetchall()]
    print(f"✓ Verified IDs in database: {db_ids}")

# Main execution
print("="*60)
print("BEIR Dataset Indexing Script")
print("="*60)
print(f"Datasets to index: {', '.join(DATASETS)}")
print()

for dataset_name in DATASETS:
    try:
        index_beir_dataset(dataset_name)
    except Exception as e:
        print(f"Error indexing {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        continue

print("\n" + "="*60)
print("INDEXING COMPLETE!")
print("="*60)
print("\nYou can now run the evaluation script:")
print("  python eval/beir_eval.py")
print()

# Show table summary
print("Database tables:")
cur.execute("""
    SELECT table_name, 
           pg_size_pretty(pg_total_relation_size(quote_ident(table_name))) as size
    FROM information_schema.tables 
    WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
    ORDER BY table_name;
""")
for row in cur.fetchall():
    print(f"  - {row[0]}: {row[1]}")

cur.close()
conn.close()