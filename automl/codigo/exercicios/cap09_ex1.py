# Capítulo 9 — solução de referência (EX 1)
import numpy as np
from scipy.stats import entropy
from sklearn.datasets import load_wine

def meta(X, y):
    n, d = X.shape; c = len(np.unique(y)); p = np.bincount(y) / n
    return {"n": n, "d": d, "n/d": round(n / d, 2), "classes": c, "entropia": round(float(entropy(p) / np.log(c)), 3)}

X, y = load_wine(return_X_y=True)
rng = np.random.default_rng(0)
print("(a) original             :", meta(X, y))
print("(b) com 100 colunas ruído:", meta(np.hstack([X, rng.normal(size=(len(y), 100))]), y))
idx = rng.choice(len(y), 60, replace=False)
print("(c) amostra de 60        :", meta(X[idx], y[idx]))
