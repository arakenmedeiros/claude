import numpy as np, torch
from ssl_profundo import *

X, y, X_te, y_te = carregar_mnist()
PASSOS = 1500
print(f"FixMatch no MNIST (100 rótulos, 10.000 sem rótulo, {PASSOS} passos, 1 sorteio): o limiar de confiança")
print("limiar | alvos que passam | precisão dos alvos | acerto do aluno")
idx = sortear_rotulados(y, 10, 0); u = np.random.RandomState(100).choice(np.setdiff1d(np.arange(len(X)), idx), 10000, replace=False)
for lim in [0.0, 0.5, 0.8, 0.95, 0.99]:
    _, _, h = treinar_ssl("fixmatch", X[idx], y[idx], X[u], passos=PASSOS, limiar=lim, yu=y[u], semente=0, X_teste=X_te, y_teste=y_te, relatar=(PASSOS,))
    print(f"{lim:6.2f} | {100 * h[0][3]:15.1f}% | {100 * h[0][4]:17.1f}% | {100 * h[0][1]:.1f}%")
