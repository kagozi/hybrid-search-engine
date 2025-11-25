# # fusion/adaptive_fusion.py
# import numpy as np
# from .query_analyzer import extract_features

# WEIGHTS = np.array([0.8, 2.0, 1.5, 0.0])   # len, entity, acronym, idf
# BIAS = -1.0

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def get_alpha(query: str) -> float:
#     f = extract_features(query)
#     x = np.array([f["length"]/10, f["entity_ratio"], f["acronym_ratio"], f["idf_variance"]])
#     return float(sigmoid(np.dot(WEIGHTS, x) + BIAS))

# fusion/adaptive_fusion.py

"""
Adaptive fusion weight (alpha) for combining BM25 and dense scores.

We use a small regression model over query features:

  alpha(q) = sigmoid( W · x(q) + b )

where x(q) = [length_norm, entity_ratio, acronym_ratio, idf_variance].

The weights (WEIGHTS, BIAS) should be learned offline on BEIR training queries
by minimizing some loss between predicted alpha and per-query optimal alpha.
Here we just store the learned parameters and offer get_alpha().
"""

import numpy as np
from .query_analyzer import extract_features


# ---------------------------------------------------------------------------
# Learned parameters (placeholder values; to be overwritten by training script)
# ---------------------------------------------------------------------------
# Example values; replace with real trained parameters.
# WEIGHTS = np.array([
#     0.5,   # length_norm
#     1.5,   # entity_ratio
#     1.0,   # acronym_ratio
#     0.2,   # idf_variance
# ], dtype=float)
WEIGHTS = np.array([
    -0.331466,   # length_norm
    0.450918,   # entity_ratio
    1.984504,   # acronym_ratio
    -0.044573  # idf_variance
], dtype=float) 

# BIAS = -0.5
BIAS =  -1.9734344960585504


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def get_alpha(query: str, dataset_name: str = None) -> float:
    """
    Compute adaptive fusion weight alpha in [0, 1].

    Optionally apply dataset-specific scaling to keep alphas in a
    reasonable range per collection (e.g., SciFact vs TREC-COVID).
    """
    f = extract_features(query)

    # normalize query length a bit
    length_norm = f["length"] / 10.0

    x = np.array([
        length_norm,
        f["entity_ratio"],
        f["acronym_ratio"],
        f["idf_variance"],
    ], dtype=float)

    raw = float(sigmoid(float(np.dot(WEIGHTS, x) + BIAS)))

    # Optional dataset-specific bounds (tune via small grid-search per dataset)
    bounds = {
        "scifact": (0.1, 0.9),       # lean more on dense/BM25 combination
        "trec-covid": (0.3, 0.8),    # BM25 more helpful for verbose biomedical queries
    }
    if dataset_name in bounds:
        lo, hi = bounds[dataset_name]
        return lo + (hi - lo) * raw

    return raw
