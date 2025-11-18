# indexing/ingest.py
import json
from bs4 import BeautifulSoup
import spacy
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    print("Downloading en_core_web_sm...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def clean_text(raw):
    soup = BeautifulSoup(raw, "html.parser")
    text = soup.get_text()
    doc = nlp(text)
    tokens = [t.text.lower() for t in doc if not t.is_stop and not t.is_punct and not t.is_space]
    return " ".join(tokens)

docs = []
input_path = "data/arxiv_docs.jsonl"
output_path = "data/arxiv_clean.jsonl"

print(f"Loading {input_path}...")
with open(input_path) as f:
    for line in tqdm(f, desc="Cleaning", unit="doc"):
        d = json.loads(line)
        d["clean_text"] = clean_text(d["text"])
        docs.append(d)

print(f"Saving {len(docs)} cleaned docs to {output_path}...")
with open(output_path, "w") as f:
    for d in docs:
        f.write(json.dumps(d) + "\n")

print("Cleaning complete!")