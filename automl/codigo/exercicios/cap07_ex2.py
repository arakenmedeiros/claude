# Capítulo 7 — solução de referência (EX 2)
import numpy as np
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold

X, y = load_digits(return_X_y=True); X = X / 16.0
cv = StratifiedKFold(3, shuffle=True, random_state=0)
limites = np.array([[-2, 3], [-5, 0]])

def nota(ind):
    return cross_val_score(SVC(C=10 ** ind[0], gamma=10 ** ind[1]), X, y, cv=cv).mean()

def ga(semente, pop=8, geracoes=5):
    rng = np.random.default_rng(semente)
    P = rng.uniform(limites[:, 0], limites[:, 1], (pop, 2)); aval = 0; melhor = 0
    for g in range(geracoes + 1):
        f = np.array([nota(i) for i in P]); aval += pop; melhor = max(melhor, f.max())
        if g == geracoes: break
        nova = [P[f.argmax()].copy()]                                         # elitismo de 1
        while len(nova) < pop:
            a, b = (P[max(rng.choice(pop, 3, replace=False), key=lambda i: f[i])] for _ in range(2))
            w = rng.uniform(0, 1); filho = w * a + (1 - w) * b                # cruzamento por média ponderada
            filho = filho + rng.normal(0, 0.3, 2)                             # mutação gaussiana
            nova.append(np.clip(filho, limites[:, 0], limites[:, 1]))
        P = np.array(nova)
    return melhor, aval

def aleatoria(semente, n):
    rng = np.random.default_rng(100 + semente)
    return max(nota(rng.uniform(limites[:, 0], limites[:, 1])) for _ in range(n))

res = []
for s in range(5):
    m, n = ga(s); res.append((m, aleatoria(s, n)))
r = np.array(res)
print("semente | genético | aleatória")
for s, (a, b) in enumerate(r): print(f"{s:7d} | {a:8.3f} | {b:9.3f}")
print(f"média   | {r[:,0].mean():8.3f} | {r[:,1].mean():9.3f}   (avaliações por método: {n})")
