import numpy as np
from fedlib import *


def epsilon_rdp(z, rodadas, delta=1e-5):
    """Limite superior de epsilon para `rodadas` aplicações do mecanismo gaussiano com multiplicador de ruído z,
    com participação total (sem amplificação por sorteio). Usa a contabilidade de Rényi (Mironov, 2017)."""
    if z <= 0:
        return float("inf")
    alfas = np.arange(1.01, 400, 0.01)
    eps = rodadas * alfas / (2 * z ** 2) + np.log(1 / delta) / (alfas - 1)
    return float(eps.min())


def fedavg_dp(partes, X, y, Xte, yte, z=1.0, S=1.0, rodadas=30, epocas=5, eta=0.1, lote=32, seed=0, classes=10):
    """FedAvg com privacidade diferencial no nível do cliente: corta cada atualização à norma S e soma ruído gaussiano."""
    rng = np.random.default_rng(seed)
    W = np.zeros((X.shape[1], classes))
    m = sum(1 for p in partes if len(p) > 0)
    historico = []
    for r in range(rodadas):
        soma = np.zeros_like(W)
        for c, p in enumerate(partes):
            if len(p) == 0:
                continue
            d = treino_local(W, X[p], y[p], epocas, eta, lote, 0.0, seed * 1000 + r * 50 + c) - W
            norma = np.linalg.norm(d)
            soma += d * min(1.0, S / (norma + 1e-12))                   # corte (clipping)
        ruido = rng.normal(0.0, z * S, W.shape)                          # ruído calibrado pela sensibilidade S
        W = W + (soma + ruido) / m
        historico.append(acuracia(W, Xte, yte))
    return W, historico
