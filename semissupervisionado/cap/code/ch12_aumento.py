import numpy as np, torch
from contrastivo import *

X, y, X_te, y_te = carregar_mnist()
Xu = X[np.random.RandomState(100).choice(len(X), 10000, replace=False)]
print("SimCLR no MNIST (1.500 passos): o aumento de dados usado para criar as duas visões. Sonda linear com 100 rótulos, 3 sorteios")
print("aumento nas visões                       | acerto no teste")
for nome, rotulo, aug in [("simclr_nenhum", "nenhum (duas cópias idênticas)", lambda x: x), ("simclr_fraco", "fraco (deslocar até 8%)", aumento_fraco), ("simclr_forte", "forte (gira, escala, apaga, ruído)", aumento_forte)]:
    enc = carregar_ou_treinar(nome, lambda: pre_treinar(Xu, "simclr", passos=1500, lote=128, aug=aug))
    v = [sonda_linear(enc, X[sortear_rotulados(y, 10, s)], y[sortear_rotulados(y, 10, s)], X_te, y_te) for s in range(3)]
    print(f"{rotulo:40s} | {100 * np.mean(v):.1f}% ± {100 * np.std(v):.1f}")
