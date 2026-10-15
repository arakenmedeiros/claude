import numpy as np, torch
from ssl_profundo import *

X, y, X_te, y_te = carregar_fashion()
espelho = lambda x: torch.where(torch.rand(len(x), 1, 1, 1) < 0.5, x.flip(3), x)              # espelhamento horizontal, com chance de 50%
opcoes = [("nenhum", None), ("deslocar até 8% (fraco)", aumento_fraco), ("só espelhar", espelho),
          ("moderado (gira 10°, escala 10%, desloca 10%)", lambda x: aumentar(x, rotacao=10, escala=0.1, deslocamento=0.1)),
          ("moderado + espelhar", lambda x: espelho(aumentar(x, rotacao=10, escala=0.1, deslocamento=0.1))),
          ("forte do Cap. 9 (gira 20°, apaga, ruído)", aumento_forte)]
print("Fashion-MNIST, CNN só com 100 rótulos (10 por classe), 1.500 passos, 3 sorteios: o aumento de dados")
print("aumento                                       | acerto no teste")
for nome, aug in opcoes:
    v = []
    for s in range(3):
        idx = sortear_rotulados(y, 10, s)
        if aug is None:
            torch.manual_seed(s); m = CNN(); v.append(treinar_supervisionado(m, X[idx], y[idx], epocas=375, aug=None, semente=s, X_teste=X_te, y_teste=y_te, relatar=(375,))[0][2])
        else: v.append(treinar_ssl("sup", X[idx], y[idx], X[:10], passos=1500, forte=aug, semente=s, X_teste=X_te, y_teste=y_te, relatar=(1500,))[2][0][1])
    print(f"{nome:45s} | {100 * np.mean(v):.1f}% ± {100 * np.std(v):.1f}")
