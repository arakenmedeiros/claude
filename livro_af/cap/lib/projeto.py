import numpy as np
from fedlib import *
from compressao import quantizar
from robustez import agg_media, agg_mediana, agg_media_aparada, com_gatilho
from personalizacao import dividir_local, acuracias_locais

AGREGADORES = {"media": agg_media, "mediana": agg_mediana, "aparada": lambda D: agg_media_aparada(D, 0.25)}


def fedavg_integrado(treino, X, y, Xte, yte, agregador="media", bits=None, dp_z=None, S=0.5,
                     maliciosos=(), gama=1.0, alvo=0, rodadas=30, epocas=5, eta=0.1, lote=32, seed=0, classes=10):
    """Motor do projeto final: junta agregador robusto, compressão por quantização, DP no nível do cliente
    (corte + ruído) e clientes maliciosos com backdoor. Devolve o modelo, a acurácia, o ASR e os bytes de subida."""
    rng = np.random.default_rng(seed)
    W = np.zeros((X.shape[1], classes))
    n_par = W.size
    maliciosos = set(maliciosos)
    Xte_gat = com_gatilho(Xte[yte != alvo])
    bytes_subida = 0
    for r in range(rodadas):
        deltas = []
        for c, p in enumerate(treino):
            if len(p) == 0:
                continue
            Xc, yc = X[p], y[p]
            if c in maliciosos:                                           # backdoor: metade extra com gatilho
                n_g = len(p) // 2
                Xc = np.vstack([Xc, com_gatilho(Xc[:n_g])]); yc = np.r_[yc, np.full(n_g, alvo)]
            d = treino_local(W, Xc, yc, epocas, eta, lote, 0.0, seed * 1000 + r * 50 + c) - W
            if c in maliciosos:
                d = gama * d
            if dp_z is not None:                                          # corte à norma S
                d = d * min(1.0, S / (np.linalg.norm(d) + 1e-12))
            if bits is not None:
                d = quantizar(d, bits, rng); bytes_subida += int(np.ceil(n_par * bits / 8)) + 8
            else:
                bytes_subida += 4 * n_par
            deltas.append(d)
        D = np.array(deltas)
        if dp_z is not None:
            media = D.mean(axis=0) + rng.normal(0, dp_z * S / len(D), W.shape)    # DP: ruído na média
            W = W + media
        else:
            W = W + AGREGADORES[agregador](D)
    acc = acuracia(W, Xte, yte)
    asr = float(np.mean(np.argmax(Xte_gat @ W, axis=1) == alvo))
    return W, acc, asr, bytes_subida
