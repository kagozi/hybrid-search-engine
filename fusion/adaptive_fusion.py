# fusion/adaptive_fusion.py
"""
Improved adaptive fusion with enhanced features and dataset-specific calibration.
"""

import numpy as np
from .query_analyzer import extract_features


# Learned parameters from improved training script
# Replace these after running train_alpha_improved.py
WEIGHTS = np.array([
    -0.013306,
    -0.386299,
    -0.026992,
    0.224702,
    -0.069304,
    -0.936562,
    0.936562,
    0.000000,
], dtype=float)

BIAS = -1.530800


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        exp_x = np.exp(x)
        return exp_x / (1.0 + exp_x)


def extract_enhanced_features(query: str, dataset_name: str = None):
    """Extract features with dataset indicators."""
    base = extract_features(query)
    
    is_scifact = 1.0 if dataset_name == "scifact" else 0.0
    is_trec = 1.0 if dataset_name == "trec-covid" else 0.0
    is_nfcorpus = 1.0 if dataset_name == "nfcorpus" else 0.0
    
    return np.array([
        base["length"] / 10.0,
        base["content_length"] / 10.0,
        base["entity_ratio"],
        base["acronym_ratio"],
        base["idf_variance"] / 10.0,
        is_scifact,
        is_trec,
        is_nfcorpus,
    ], dtype=float)


def get_alpha(query: str, dataset_name: str = None) -> float:
    """
    Compute adaptive fusion weight alpha ∈ [0, 1].
    
    Higher alpha → more BM25 weight
    Lower alpha → more dense weight
    
    Args:
        query: Query text
        dataset_name: Optional dataset name for calibration
    
    Returns:
        Alpha weight for BM25 in [0, 1]
    """
    x = extract_enhanced_features(query, dataset_name)
    logit = float(np.dot(WEIGHTS, x) + BIAS)
    raw_alpha = sigmoid(logit)
    
    # Dataset-specific post-calibration
    # These bounds help prevent extreme weights and improve robustness
    if dataset_name == "scifact":
        # SciFact: Dense models work very well, keep alpha moderate
        return np.clip(raw_alpha, 0.15, 0.75)
    elif dataset_name == "trec-covid":
        # TREC-COVID: More lexical matching helps, allow higher BM25 weight
        return np.clip(raw_alpha, 0.25, 0.85)
    elif dataset_name == "nfcorpus":
        # NFCorpus: Medical/nutrition queries, balanced approach
        return np.clip(raw_alpha, 0.20, 0.80)
    else:
        # Default: balanced range
        return np.clip(raw_alpha, 0.2, 0.8)


def get_alpha_ensemble(query: str, dataset_name: str = None, methods=None):
    """
    Get multiple alpha values for ensemble fusion.
    
    Returns dict with different alpha strategies:
    - 'adaptive': Learned alpha from features
    - 'fixed_balanced': 0.5 (equal weight)
    - 'bm25_heavy': 0.7 (favor BM25)
    - 'dense_heavy': 0.3 (favor dense)
    """
    if methods is None:
        methods = ['adaptive', 'fixed_balanced', 'bm25_heavy', 'dense_heavy']
    
    alphas = {}
    
    if 'adaptive' in methods:
        alphas['adaptive'] = get_alpha(query, dataset_name)
    
    if 'fixed_balanced' in methods:
        alphas['fixed_balanced'] = 0.5
    
    if 'bm25_heavy' in methods:
        if dataset_name == "trec-covid":
            alphas['bm25_heavy'] = 0.7
        elif dataset_name == "nfcorpus":
            alphas['bm25_heavy'] = 0.65
        else:
            alphas['bm25_heavy'] = 0.65
    
    if 'dense_heavy' in methods:
        if dataset_name == "scifact":
            alphas['dense_heavy'] = 0.25
        elif dataset_name == "nfcorpus":
            alphas['dense_heavy'] = 0.30
        else:
            alphas['dense_heavy'] = 0.3
    
    return alphas