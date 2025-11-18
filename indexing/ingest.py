# indexing/ingest.py
import json
from bs4 import BeautifulSoup
import spacy
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

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

input_path = "data/arxiv_docs.jsonl"
output_path = "data/arxiv_clean.jsonl"

docs = []
print(f"Loading {input_path}...")
with open(input_path) as f:
    for line in tqdm(f, desc="Cleaning", unit="doc"):
        d = json.loads(line.strip())
        # Use abstract as main text (not "text")
        full_text = d.get("abstract", "") or ""
        if "title" in d:
            full_text = d["title"] + " " + full_text  # include title for better retrieval
        
        d["text"] = full_text.strip()           # keep original
        d["clean_text"] = clean_text(full_text)  # cleaned version
        
        # Clean up authors if needed
        if isinstance(d.get("authors"), list):
            d["authors"] = [a.strip() for a in d["authors"] if a.strip() and a.strip() != "\n"]
        
        docs.append(d)

print(f"Saving {len(docs)} cleaned docs to {output_path}...")
with open(output_path, "w") as f:
    for d in docs:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")

print("Cleaning complete! Ready for indexing.")