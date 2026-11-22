# Capítulo 11 — solução de referência (EX 3)
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
Xa, Xv, ya, yv = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
sc = StandardScaler().fit(Xa); Xa, Xv = sc.transform(Xa), sc.transform(Xv)
LARG, ATIV = [4, 8, 16, 32, 64], ["relu", "tanh"]
cache = {}
def nota(a):
    if a not in cache: cache[a] = MLPClassifier(a[0], activation=a[1], max_iter=100, random_state=0).fit(Xa, ya).score(Xv, yv)
    return cache[a]
def sorteia(r):
    return (tuple(int(r.choice(LARG)) for _ in range(int(r.integers(1, 4)))), str(r.choice(ATIV)))
def muta(a, r):
    h, at = list(a[0]), a[1]; op = int(r.integers(4))
    if op == 0: h[int(r.integers(len(h)))] = int(r.choice(LARG))
    elif op == 1 and len(h) < 3: h.insert(int(r.integers(len(h) + 1)), int(r.choice(LARG)))
    elif op == 2 and len(h) > 1: h.pop(int(r.integers(len(h))))
    else: at = "tanh" if at == "relu" else "relu"
    return (tuple(h), at)
B = 15; R = {"aleatória": [], "hill climbing": []}
for s in range(5):
    r = np.random.default_rng(s)
    R["aleatória"].append(max(nota(sorteia(r)) for _ in range(B)))
    r = np.random.default_rng(100 + s); atual = sorteia(r); melhor = nota(atual)
    for _ in range(B - 1):
        f = muta(atual, r)
        if nota(f) > melhor: atual, melhor = f, nota(f)
    R["hill climbing"].append(melhor)
for k, v in R.items(): print(f"{k:<14s} média = {np.mean(v):.4f}  desvio = {np.std(v):.4f}  valores = {np.round(v, 3).tolist()}")
