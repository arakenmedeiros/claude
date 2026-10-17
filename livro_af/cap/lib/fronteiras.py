import numpy as np
from fedlib import *


# ------------------------------------------------------------------ K-means federado
def kmeans_federado(partes, X, centroides, iteracoes=30):
    """Algoritmo de Lloyd distribuído: cada cliente devolve, por grupo, a soma dos pontos e a contagem."""
    C = centroides.copy()
    for _ in range(iteracoes):
        S, N = np.zeros_like(C), np.zeros(len(C))
        for p in partes:                                   # (em um sistema real, isto roda em cada cliente)
            Xk = X[p]
            a = ((Xk[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
            for j in range(len(C)):
                S[j] += Xk[a == j].sum(0)
                N[j] += (a == j).sum()
        C = np.where(N[:, None] > 0, S / np.maximum(N, 1)[:, None], C)   # servidor: nova média de cada grupo
    return C


# ------------------------------------------------------------------ PCA federado
def pca_federado(partes, X, n_componentes):
    """PCA exato a partir de somas locais: cada cliente envia n, Σx e Σxxᵀ."""
    n, s, ss = 0, 0.0, 0.0
    for p in partes:
        Xk = X[p]
        n += len(Xk); s = s + Xk.sum(0); ss = ss + Xk.T @ Xk
    mu = s / n
    cov = (ss - n * np.outer(mu, mu)) / (n - 1)
    valores, vetores = np.linalg.eigh(cov)
    ordem = np.argsort(valores)[::-1][:n_componentes]
    return mu, valores[ordem], vetores[:, ordem]


# ------------------------------------------------------------------ Adaptadores de baixo posto (estilo LoRA)
def fedavg_adaptador(partes, X, y, Xte, yte, W0, posto, rodadas=20, epocas=3, eta=0.1, lote=32, seed=0):
    """Mantém W0 congelado e treina só ΔW = A·B, com A (d×r) e B (r×K). Só A e B trafegam.
    Com posto=None, treina o modelo inteiro (referência)."""
    rng = np.random.default_rng(seed)
    d, K = W0.shape
    A = rng.normal(0, 0.01, (d, posto)) if posto else None
    B = np.zeros((posto, K)) if posto else None
    W = W0.copy()
    bytes_por_rodada = 2 * 4 * (d * K if posto is None else posto * (d + K))
    historico = []
    for r in range(rodadas):
        locais, tam = [], []
        for c, p in enumerate(partes):
            if len(p) == 0:
                continue
            rk = np.random.default_rng(seed * 1000 + r * 50 + c)
            if posto is None:
                w = W.copy()
                for _ in range(epocas):
                    o = rk.permutation(len(p))
                    for i in range(0, len(p), lote):
                        b = p[o[i:i + lote]]; w -= eta * gradiente(w, X[b], y[b])
                locais.append(w)
            else:
                a, bb = A.copy(), B.copy()
                for _ in range(epocas):
                    o = rk.permutation(len(p))
                    for i in range(0, len(p), lote):
                        b = p[o[i:i + lote]]
                        G = gradiente(W0 + a @ bb, X[b], y[b])           # gradiente em relação a ΔW
                        a, bb = a - eta * G @ bb.T, bb - eta * a.T @ G   # regra da cadeia para A e B
                locais.append((a, bb))
            tam.append(len(p))
        pesos = np.array(tam) / sum(tam)
        if posto is None:
            W = sum(pi * wi for pi, wi in zip(pesos, locais)); acc = acuracia(W, Xte, yte)
        else:
            A = sum(pi * l[0] for pi, l in zip(pesos, locais)); B = sum(pi * l[1] for pi, l in zip(pesos, locais))
            acc = acuracia(W0 + A @ B, Xte, yte)
        historico.append(acc)
    return historico, bytes_por_rodada
