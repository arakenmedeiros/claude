import numpy as np
from fedlib import *


def quantizar(delta, bits, rng):
    """Quantização uniforme com arredondamento estocástico (não enviesada)."""
    lo, hi = delta.min(), delta.max()
    if hi == lo:
        return delta.copy()
    niveis = 2 ** bits - 1
    z = (delta - lo) / (hi - lo) * niveis            # leva os valores para [0, niveis]
    base = np.floor(z)
    z = base + (rng.random(z.shape) < (z - base))    # sobe com probabilidade igual à parte fracionária
    return lo + z / niveis * (hi - lo)


def top_k(delta, frac):
    """Mantém apenas os k maiores valores (em módulo); zera o resto."""
    k = max(1, int(round(frac * delta.size)))
    corte = np.sort(np.abs(delta).ravel())[-k]
    return np.where(np.abs(delta) >= corte, delta, 0.0)


def bytes_quantizado(n, bits):
    return int(np.ceil(n * bits / 8)) + 8            # +8: mínimo e máximo em 32 bits cada


def bytes_esparso(n, frac):
    k = max(1, int(round(frac * n)))
    return k * (4 + 2)                               # valor (4 bytes) + índice (2 bytes)


def fedavg_comprimido(partes, X, y, Xte, yte, metodo="nenhum", param=None, erro_acumulado=False,
                      rodadas=30, epocas=5, eta=0.1, lote=32, frac=0.5, seed=0, classes=10):
    """FedAvg em que cada cliente comprime a sua atualização (delta) antes de enviá-la."""
    rng = np.random.default_rng(seed)
    W = np.zeros((X.shape[1], classes))
    n = W.size
    memoria = [np.zeros_like(W) for _ in partes]     # erro acumulado de cada cliente
    total_subida, historico = 0, []
    for r in range(rodadas):
        m = max(1, int(round(frac * len(partes))))
        deltas, tam = [], []
        for c in rng.choice(len(partes), m, replace=False):
            if len(partes[c]) == 0:
                continue
            Wc = treino_local(W, X[partes[c]], y[partes[c]], epocas, eta, lote, 0.0, seed * 1000 + r * 50 + int(c))
            d = Wc - W
            if erro_acumulado:
                d = d + memoria[c]
            if metodo == "quantizar":
                dh, b = quantizar(d, param, rng), bytes_quantizado(n, param)
            elif metodo == "topk":
                dh, b = top_k(d, param), bytes_esparso(n, param)
            else:
                dh, b = d, 4 * n
            if erro_acumulado:
                memoria[c] = d - dh
            deltas.append(dh); tam.append(len(partes[c])); total_subida += b
        p = np.array(tam) / sum(tam)
        W = W + sum(pi * di for pi, di in zip(p, deltas))
        historico.append(acuracia(W, Xte, yte))
    return W, historico, total_subida
