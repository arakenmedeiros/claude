import numpy as np
from avancado import *

Xbr, Xteb, ytr, yte = carregar_digits(bruto=True)
K, NOVOS = 30, 6
R = {}
cenarios = {'sem estilo': 0.0, 'estilo moderado': 1.0, 'estilo forte': 2.0,
            'novos de outra população': np.r_[np.full(K - NOVOS, 1.0), np.full(NOVOS, 2.5)]}
for nome, intens in cenarios.items():
    for s in range(5):
        partes = particao_dirichlet(ytr, K, 0.5, s)
        Xs, ganho, desloc = federacao_com_estilo(Xbr, partes, seed=s, intensidade=intens)
        conhecidos, novos = list(range(K - NOVOS)), list(range(K - NOVOS, K))      # os 6 últimos nunca participam do treino
        treino, teste = dividir_local([partes[c] for c in conhecidos], 0.3, s)
        pad = ajustar_padronizador(Xs, treino)       # aprendido só com os clientes conhecidos
        Xall = pad(Xs)
        W, _ = fedavg(treino, Xall, ytr, pad(Xteb), yte, rodadas=30, epocas=5, seed=s)
        acc_conh = np.array([acuracia(W, Xall[t], ytr[t]) for t in teste])
        acc_novo = np.array([acuracia(W, Xall[partes[c]], ytr[partes[c]]) for c in novos if len(partes[c]) >= 5])
        R.setdefault(nome, []).append((acc_conh.mean(), np.percentile(acc_conh, 10), acc_novo.mean(), acc_novo.min()))
print(f"{'cenário':<26}{'conhecidos: média':>18}{'p10':>7}{'novos: média':>14}{'pior novo':>11}")
for nome, v in R.items():
    m = np.mean(v, axis=0)
    print(f"{nome:<26}{m[0]:>18.3f}{m[1]:>7.3f}{m[2]:>14.3f}{m[3]:>11.3f}")
