import numpy as np
from avancado import *

Xtr, Xte, ytr, yte = carregar_digits()
K = 20
R = {}
for s in range(5):
    rng = np.random.default_rng(s)
    partes = particao_dirichlet(ytr, K, 0.5, s)
    rot = set(rng.choice(K, 3, replace=False).tolist())                  # só 3 clientes têm rótulos
    for nome, lim in (("só rotulados", 2.0), ("pseudorrótulos (limiar 0,5)", 0.5), ("pseudorrótulos (limiar 0,8)", 0.8), ("pseudorrótulos (limiar 0,95)", 0.95)):
        _, acc, prec = fedavg_pseudorotulos(partes, rot, Xtr, ytr, Xte, yte, limiar=lim, seed=s)
        R.setdefault(nome, []).append((acc, prec))
    R.setdefault("todos rotulados (teto)", []).append((fedavg_pseudorotulos(partes, set(range(K)), Xtr, ytr, Xte, yte, seed=s)[1], np.nan))
print(f"{'estratégia':<32}{'acurácia':>9}{'precisão dos pseudorrótulos':>29}")
for nome, v in R.items():
    m = np.nanmean(v, axis=0)
    print(f"{nome:<32}{m[0]:>9.3f}{('—' if np.isnan(m[1]) else f'{m[1]:.3f}'):>29}")
