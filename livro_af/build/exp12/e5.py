import numpy as np
from avancado import *

Xbr, Xteb, ytr, yte = carregar_digits(bruto=True)
K = 10
print(f"{'estilos':>10}{'método':>10}{'média por cliente':>19}{'percentil 10':>14}")
for intens in (0.0, 1.0):
    for modo in ("fedavg", "fedbn"):
        acc = []
        for s in range(3):
            partes = particao_iid(ytr, K, s)
            Xs, _, _ = federacao_com_estilo(Xbr, partes, seed=s, intensidade=intens)
            treino, teste = dividir_local(partes, 0.3, s)
            mu = np.vstack([Xs[p] for p in treino]).mean(0); sd = np.vstack([Xs[p] for p in treino]).std(0); sd[sd == 0] = 1
            Xn = (Xs - mu) / sd
            prever = fedbn_treinar([Xn[p] for p in treino], [ytr[p] for p in treino], modo, seed=s)
            acc += [np.mean(prever(c, Xn[teste[c]]) == ytr[teste[c]]) for c in range(K)]
        print(f"{intens:>10}{modo:>10}{np.mean(acc):>19.3f}{np.percentile(acc, 10):>14.3f}")
