import numpy as np, torch
from ssl_profundo import *

X, y, X_te, y_te = carregar_mnist()
PASSOS = 1500
print(f"MNIST, 10.000 sem rótulo, {PASSOS} passos, 1 sorteio: e com ainda menos rótulos?")
print("rótulos (por dígito) | só rotulados | Pseudo-Label | FixMatch (0,95) | precisão dos alvos do FixMatch")
for por in [1, 3, 10]:
    idx = sortear_rotulados(y, por, 0); u = np.random.RandomState(100).choice(np.setdiff1d(np.arange(len(X)), idx), 10000, replace=False)
    r = {}
    for m in ["sup", "pseudo", "fixmatch"]:
        _, _, h = treinar_ssl(m, X[idx], y[idx], X[u], passos=PASSOS, yu=y[u], semente=0, X_teste=X_te, y_teste=y_te, relatar=(PASSOS,)); r[m] = h[0]
    print(f"{10 * por:4d} ({por:2d})           | {100 * r['sup'][1]:11.1f}% | {100 * r['pseudo'][1]:11.1f}% | {100 * r['fixmatch'][1]:14.1f}% | {100 * r['fixmatch'][4]:.1f}%")
