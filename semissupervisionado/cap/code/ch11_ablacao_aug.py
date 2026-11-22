import numpy as np, torch
from ssl_profundo import *

X, y, X_te, y_te = carregar_mnist()
PASSOS, SEMENTES = 1500, 3
print(f"MNIST, 100 rótulos + 10.000 sem rótulo, {PASSOS} passos, {SEMENTES} sorteios: separando o efeito do aumento do efeito dos pseudo-rótulos")
print("configuração                                              | aluno (teste)")
cfg = [("só rotulados, aumento forte [referência dos Capítulos 10 a 18]", "sup", {}),
       ("só rotulados, aumento fraco", "sup", dict(forte=aumento_fraco)),
       ("Pseudo-Label (rotulados com aumento fraco) [o do Capítulo 11]", "pseudo", {}),
       ("Pseudo-Label com aumento FORTE nos rotulados", "pseudo", dict(aug_sup=aumento_forte))]
for nome, m, kw in cfg:
    v = []
    for s in range(SEMENTES):
        idx = sortear_rotulados(y, 10, s); u = np.random.RandomState(100 + s).choice(np.setdiff1d(np.arange(len(X)), idx), 10000, replace=False)
        v.append(treinar_ssl(m, X[idx], y[idx], X[u], passos=PASSOS, semente=s, X_teste=X_te, y_teste=y_te, relatar=(PASSOS,), **kw)[2][0][1])
    print(f"{nome:57s} | {100 * np.mean(v):.1f}% ± {100 * np.std(v):.1f}")
