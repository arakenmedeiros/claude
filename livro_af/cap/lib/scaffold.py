import numpy as np
from fedlib import *

def scaffold(partes, X, y, Xte, yte, rodadas=40, epocas=10, eta=0.1, lote=32,
             frac=0.25, seed=0, classes=10):
    rng = np.random.default_rng(seed)
    W = np.zeros((X.shape[1], classes))
    c = np.zeros_like(W)                          # controle do servidor
    ck = [np.zeros_like(W) for _ in partes]       # controle de cada cliente
    historico = []
    for r in range(rodadas):
        m = max(1, int(round(frac * len(partes))))
        escolhidos = rng.choice(len(partes), m, replace=False)
        dW, dC = [], []
        for k in escolhidos:
            idx = partes[k]
            if len(idx) == 0:
                continue
            Xk, yk = X[idx], y[idx]
            w, passos = W.copy(), 0
            rk = np.random.default_rng(seed * 1000 + r * 50 + int(k))
            for _ in range(epocas):
                ordem = rk.permutation(len(yk))
                for i in range(0, len(yk), lote):
                    b = ordem[i:i + lote]
                    w -= eta * (gradiente(w, Xk[b], yk[b]) - ck[k] + c)   # passo corrigido
                    passos += 1
            ck_novo = ck[k] - c + (W - w) / (passos * eta)               # novo controle local
            dW.append(w - W)
            dC.append(ck_novo - ck[k])
            ck[k] = ck_novo
        W = W + np.mean(dW, axis=0)
        c = c + (len(dW) / len(partes)) * np.mean(dC, axis=0)
        historico.append(acuracia(W, Xte, yte))
    return W, historico
