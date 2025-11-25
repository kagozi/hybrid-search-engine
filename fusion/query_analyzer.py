# fusion/query_analyzer.py
"""
Query feature extraction for adaptive fusion.

We extract cheap, interpretable features from the query text:
  - length: total tokens
  - content_length: non-stopword tokens
  - entity_ratio: named entities / tokens
  - acronym_ratio: ALL-CAPS tokens / tokens
  - idf_variance: variance of content token lengths (proxy for lexical variability)
"""

import re
import math
import spacy

try:
    # keep tagger so entities & basic POS work better; parser not needed
    nlp = spacy.load("en_core_web_sm", disable=["parser"])
except OSError:
    print("Warning: en_core_web_sm not found. Using blank model.")
    nlp = spacy.blank("en")


def extract_features(q: str):
    """Return a dict of query features for adaptive fusion."""
    if not q:
        return {
            "length": 0,
            "content_length": 0,
            "entity_ratio": 0.0,
            "acronym_ratio": 0.0,
            "idf_variance": 0.0,
        }

    doc = nlp(q)
    tokens = [t.text for t in doc if not t.is_space and not t.is_punct]
    if not tokens:
        return {
            "length": 0,
            "content_length": 0,
            "entity_ratio": 0.0,
            "acronym_ratio": 0.0,
            "idf_variance": 0.0,
        }

    # content tokens = non-stopwords
    stopwords = nlp.Defaults.stop_words
    content_tokens = [t for t in tokens if t.lower() not in stopwords]

    length = len(tokens)
    content_length = len(content_tokens)

    # entity ratio
    entity_ratio = len(doc.ents) / max(length, 1)

    # acronym ratio: ALLCAPS words (at least 2 chars)
    acronym_ratio = len(re.findall(r"\b[A-Z]{2,}\b", q)) / max(length, 1)

    # "idf variance" proxy: variance of content token lengths
    if content_tokens:
        token_lens = [len(t) for t in content_tokens]
    else:
        token_lens = [len(t) for t in tokens]

    mean_len = sum(token_lens) / len(token_lens)
    idf_var = sum((l - mean_len) ** 2 for l in token_lens) / len(token_lens)

    return {
        "length": length,
        "content_length": content_length,
        "entity_ratio": float(entity_ratio),
        "acronym_ratio": float(acronym_ratio),
        "idf_variance": float(idf_var),
    }
