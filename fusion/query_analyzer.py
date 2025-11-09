# fusion/query_analyzer.py
import re
import spacy

try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger"])
except OSError:
    print("Warning: en_core_web_sm not found. Using blank model.")
    nlp = spacy.blank("en")

def extract_features(q: str):
    doc = nlp(q.lower())
    tokens = [t.text for t in doc if not t.is_stop and not t.is_punct]
    return {
        "length": len(tokens),
        "entity_ratio": len(doc.ents) / max(len(tokens), 1),
        "acronym_ratio": len(re.findall(r'\b[A-Z]{2,}\b', q)) / max(len(tokens), 1),
        "idf_variance": 0.0
    }