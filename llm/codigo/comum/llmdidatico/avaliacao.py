"""Métricas de recuperação e utilitários estatísticos."""
import math
import numpy as np


def recall_at_k(ranking, relevantes, k):
    if not relevantes: return None
    return len(set(ranking[:k]) & set(relevantes)) / len(set(relevantes))


def hit_at_k(ranking, relevantes, k):
    if not relevantes: return None
    return float(len(set(ranking[:k]) & set(relevantes)) > 0)


def mrr(ranking, relevantes):
    if not relevantes: return None
    for i, d in enumerate(ranking, 1):
        if d in relevantes: return 1.0 / i
    return 0.0


def ndcg_at_k(ranking, relevantes, k):
    if not relevantes: return None
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ranking[:k]) if d in relevantes)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevantes))))
    return dcg / idcg


def bootstrap_ic(valores, n=2000, alpha=0.05, seed=0):
    v = np.asarray(valores, float); rng = np.random.default_rng(seed)
    medias = [rng.choice(v, len(v), replace=True).mean() for _ in range(n)]
    return float(v.mean()), float(np.quantile(medias, alpha / 2)), float(np.quantile(medias, 1 - alpha / 2))
