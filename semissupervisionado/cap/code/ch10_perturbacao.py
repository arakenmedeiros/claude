import numpy as np, torch
from ssl_profundo import *

X, y, X_te, y_te = carregar_mnist()
PASSOS, SEMENTES = 1500, 2
print(f"Mean Teacher no MNIST (100 rótulos + 10.000 sem rótulo, {PASSOS} passos, {SEMENTES} sorteios): qual perturbação nos exemplos sem rótulo?")
print("perturbação (só nos sem rótulo)        | aluno (teste) | professor EMA (teste)")
for nome, aug in [("nenhuma (só dropout)", lambda x: x), ("fraca (deslocar até 8%)", aumento_fraco), ("forte (gira, escala, apaga, ruído)", aumento_forte)]:
    a, p = [], []
    for s in range(SEMENTES):
        idx = sortear_rotulados(y, 10, s)
        u = np.random.RandomState(100 + s).choice(np.setdiff1d(np.arange(len(X)), idx), 10000, replace=False)
        _, _, h = treinar_ssl("mt", X[idx], y[idx], X[u], passos=PASSOS, aug_u=aug, semente=s, X_teste=X_te, y_teste=y_te, relatar=(PASSOS,))
        a.append(h[0][1]); p.append(h[0][2])
    print(f"{nome:38s} | {100 * np.mean(a):12.1f}% | {100 * np.mean(p):.1f}%")
