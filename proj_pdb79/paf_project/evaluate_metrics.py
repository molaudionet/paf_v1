# evaluate_metrics.py
from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple
from collections import Counter
from scipy.stats import ttest_ind

def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))

def pairwise_similarity(keys: List[str], emb: Dict[str, np.ndarray]) -> np.ndarray:
    n = len(keys)
    M = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        M[i, i] = 1.0
        for j in range(i + 1, n):
            s = cosine(emb[keys[i]], emb[keys[j]])
            M[i, j] = s
            M[j, i] = s
    return M

def loo_knn_accuracy(keys: List[str], labels: Dict[str, str], S: np.ndarray, k: int = 1) -> float:
    n = len(keys)
    correct = 0
    for i in range(n):
        sims = S[i].copy()
        sims[i] = -1e9
        nn = np.argsort(-sims)[:k]
        vote = Counter([labels[keys[j]] for j in nn]).most_common(1)[0][0]
        if vote == labels[keys[i]]:
            correct += 1
    return correct / n

def within_between_stats(keys: List[str], labels: Dict[str, str], S: np.ndarray) -> Tuple[float, float, float]:
    """
    Returns:
      cohen_d, p_value, (mean_within - mean_between)
    """
    within = []
    between = []
    n = len(keys)
    for i in range(n):
        for j in range(i + 1, n):
            if labels[keys[i]] == labels[keys[j]]:
                within.append(float(S[i, j]))
            else:
                between.append(float(S[i, j]))
    within = np.array(within, dtype=np.float32)
    between = np.array(between, dtype=np.float32)
    # Cohen's d
    mw, mb = float(within.mean()), float(between.mean())
    sw, sb = float(within.std(ddof=1) + 1e-8), float(between.std(ddof=1) + 1e-8)
    sp = np.sqrt((sw * sw + sb * sb) / 2.0)
    d = (mw - mb) / (sp + 1e-8)
    # t-test (informal; dependence exists; still useful as quick signal)
    p = float(ttest_ind(within, between, equal_var=False).pvalue)
    return float(d), p, float(mw - mb)
