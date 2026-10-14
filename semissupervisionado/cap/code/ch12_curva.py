import numpy as np, torch
from contrastivo import *

X, y, X_te, y_te = carregar_mnist()
Xu = X[np.random.RandomState(100).choice(len(X), 10000, replace=False)]
enc = carregar_ou_treinar("simclr_forte", lambda: pre_treinar(Xu, "simclr", passos=1500, lote=128))
print("MNIST: sonda linear sobre o codificador SimCLR x regressão logística sobre pixels, conforme o número de rótulos (3 sorteios)")
print("rótulos (por dígito) | pixels | sonda linear SimCLR")
for por in [1, 3, 10, 30, 100]:
    a, b = [], []
    for s in range(3):
        idx = sortear_rotulados(y, por, s)
        a.append(LogisticRegression(max_iter=3000).fit(X[idx].flatten(1).numpy(), y[idx].numpy()).score(X_te.flatten(1).numpy(), y_te.numpy()))
        b.append(sonda_linear(enc, X[idx], y[idx], X_te, y_te))
    print(f"{10 * por:5d} ({por:3d})        | {100 * np.mean(a):5.1f}% | {100 * np.mean(b):5.1f}%")
