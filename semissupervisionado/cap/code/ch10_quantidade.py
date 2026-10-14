import numpy as np, torch
from ssl_profundo import *

X, y, X_te, y_te = carregar_mnist()
PASSOS = 1500
print(f"UDA no MNIST (100 rótulos, {PASSOS} passos, 1 sorteio): quantas imagens sem rótulo? (0 = só a referência supervisionada)")
print("sem rótulo | aluno (teste) | professor EMA (teste)")
idx = sortear_rotulados(y, 10, 0); pool = np.setdiff1d(np.arange(len(X)), idx)
_, _, h = treinar_ssl("sup", X[idx], y[idx], X[pool[:10]], passos=PASSOS, semente=0, X_teste=X_te, y_teste=y_te, relatar=(PASSOS,))
print(f"{0:10d} | {100 * h[0][1]:12.1f}% | {100 * h[0][2]:.1f}%")
for n in [100, 1000, 10000, 40000]:
    u = np.random.RandomState(100).choice(pool, n, replace=False)
    _, _, h = treinar_ssl("uda", X[idx], y[idx], X[u], passos=PASSOS, semente=0, X_teste=X_te, y_teste=y_te, relatar=(PASSOS,))
    print(f"{n:10d} | {100 * h[0][1]:12.1f}% | {100 * h[0][2]:.1f}%")
