# indexing/ingest.py
import json
from bs4 import BeautifulSoup
import spacy
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def clean_text(raw):
    soup = BeautifulSoup(raw, "html.parser")
    text = soup.get_text()
    doc = nlp(text)
    tokens = [t.text.lower() for t in doc if not t.is_stop and not t.is_punct and not t.is_space]
    return " ".join(tokens)

docs = []
with open("../data/arxiv_docs.jsonl") as f:
    for line in tqdm(f, desc="Cleaning"):
        d = json.loads(line)
        d["clean_text"] = clean_text(d["text"])
        docs.append(d)

with open("../data/arxiv_clean.jsonl", "w") as f:
    for d in docs:
        f.write(json.dumps(d) + "\n")