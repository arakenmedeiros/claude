import numpy as np
from fedlib import *


# ---------- agregadores (recebem um array de atualizações com forma (m, d, classes)) ----------
def agg_media(D):
    return D.mean(axis=0)


def agg_mediana(D):
    return np.median(D, axis=0)


def agg_media_aparada(D, beta=0.2):
    m = D.shape[0]
    k = int(np.floor(beta * m))
    S = np.sort(D, axis=0)
    return S[k:m - k].mean(axis=0)


def agg_krum(D, f=2, multi=1):
    """Krum (Blanchard et al., 2017): escolhe a(s) atualização(ões) mais 'central(is)'."""
    m = D.shape[0]
    X = D.reshape(m, -1)
    dist = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    k = max(1, m - f - 2)
    scores = np.array([np.sort(np.delete(dist[i], i))[:k].sum() for i in range(m)])
    escolhidos = np.argsort(scores)[:multi]
    return D[escolhidos].mean(axis=0)


def agg_corte_norma(D, S=1.0):
    """Corta a norma de cada atualização a S e tira a média."""
    n = np.linalg.norm(D.reshape(len(D), -1), axis=1)
    fator = np.minimum(1.0, S / (n + 1e-12))
    return (D * fator[:, None, None]).mean(axis=0)


# ---------- ataques ----------
def com_gatilho(X, tamanho=2):
    """Gatilho (backdoor): acende o canto inferior direito da imagem 8x8 (nas coordenadas padronizadas)."""
    Xg = X.copy()
    img = Xg[:, :-1].reshape(-1, 8, 8)
    img[:, 8 - tamanho:, 8 - tamanho:] = 6.0
    Xg[:, :-1] = img.reshape(len(Xg), -1)
    return Xg


def fedavg_robusto(partes, X, y, Xte, yte, agregador=agg_media, maliciosos=(), ataque="nenhum", gama=3.0,
                   rodadas=30, epocas=5, eta=0.1, lote=32, seed=0, classes=10, alvo=0):
    """FedAvg com clientes maliciosos e agregador à escolha. Devolve o modelo e os históricos de acurácia e de sucesso do backdoor."""
    rng = np.random.default_rng(seed)
    W = np.zeros((X.shape[1], classes))
    maliciosos = set(maliciosos)
    Xte_gat = com_gatilho(Xte[yte != alvo])
    hist_acc, hist_asr = [], []
    for r in range(rodadas):
        deltas = []
        for c, p in enumerate(partes):
            if len(p) == 0:
                continue
            Xc, yc = X[p], y[p]
            if c in maliciosos and ataque == "rotulos":
                yc = (yc + 1) % classes                                   # troca de rótulos
            if c in maliciosos and ataque == "backdoor":
                n_g = len(p) // 2
                Xc = np.vstack([Xc, com_gatilho(Xc[:n_g])]); yc = np.r_[yc, np.full(n_g, alvo)]
            d = treino_local(W, Xc, yc, epocas, eta, lote, 0.0, seed * 1000 + r * 50 + c) - W
            if c in maliciosos and ataque == "sinal":
                d = -gama * d                                             # inverte e amplifica
            if c in maliciosos and ataque == "ruido":
                d = rng.normal(0, gama, d.shape)
            if c in maliciosos and ataque == "backdoor":
                d = gama * d                                              # amplifica (substituição de modelo)
            deltas.append(d)
        W = W + agregador(np.array(deltas))
        hist_acc.append(acuracia(W, Xte, yte))
        hist_asr.append(float(np.mean(np.argmax(Xte_gat @ W, axis=1) == alvo)))
    return W, hist_acc, hist_asr
