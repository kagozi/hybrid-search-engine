# fusion/adaptive_fusion.py
import numpy as np
from .query_analyzer import extract_features

WEIGHTS = np.array([0.8, 2.0, 1.5, 0.0])   # len, entity, acronym, idf
BIAS = -1.0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_alpha(query: str) -> float:
    f = extract_features(query)
    x = np.array([f["length"]/10, f["entity_ratio"], f["acronym_ratio"], f["idf_variance"]])
    return float(sigmoid(np.dot(WEIGHTS, x) + BIAS))