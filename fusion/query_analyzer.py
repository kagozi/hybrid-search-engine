# fusion/query_analyzer.py
import re
import spacy

nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger"])

def extract_features(q: str):
    doc = nlp(q.lower())
    tokens = [t.text for t in doc if not t.is_stop and not t.is_punct]
    return {
        "length": len(tokens),
        "entity_ratio": len(doc.ents) / max(len(tokens), 1),
        "acronym_ratio": len(re.findall(r'\b[A-Z]{2,}\b', q)) / max(len(tokens), 1),
        "idf_variance": 0.0   # placeholder
    }