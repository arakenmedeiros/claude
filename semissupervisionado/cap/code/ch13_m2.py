import numpy as np, torch
from generativos import *

X, y, X_te, y_te = carregar_mnist()
Xu = X[np.random.RandomState(100).choice(len(X), 10000, replace=False)]
PASSOS = (500, 1000, 1500)
print("VAE semissupervisionado (M2 simplificado) x o mesmo classificador treinado só com os 100 rótulos (MNIST, 3 sorteios)")
print("passos | M2: acerto do classificador q(y|x) | só rotulados (mesma rede, sem aumento)")
M, R = [], []
for s in range(3):
    idx = sortear_rotulados(y, 10, s)
    _, h = treinar_m2(X[idx], y[idx], Xu, passos=PASSOS[-1], semente=s, X_teste=X_te, y_teste=y_te, relatar=PASSOS); M.append([a for _, a in h])
    R.append([treinar_mlp(X[idx], y[idx], passos=p, semente=s, X_teste=X_te, y_teste=y_te) for p in PASSOS])
M, R = np.array(M), np.array(R)
for i, p in enumerate(PASSOS): print(f"{p:6d} | {100 * M[:, i].mean():18.1f}% ± {100 * M[:, i].std():.1f}        | {100 * R[:, i].mean():.1f}% ± {100 * R[:, i].std():.1f}")
