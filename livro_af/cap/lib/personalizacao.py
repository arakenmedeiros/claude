import numpy as np
from fedlib import *


def dividir_local(partes, frac_teste=0.3, seed=0):
    """Cada cliente separa uma parte dos seus próprios dados para teste local."""
    rng = np.random.default_rng(seed)
    treino, teste = [], []
    for p in partes:
        q = rng.permutation(p)
        k = int(len(q) * (1 - frac_teste))
        treino.append(q[:k]); teste.append(q[k:])
    return treino, teste


def acuracias_locais(modelos, teste, X, y):
    """Acurácia de cada cliente, usando o modelo que cabe a ele e o seu teste local."""
    return np.array([acuracia(W, X[t], y[t]) for W, t in zip(modelos, teste)])


def resumo(acc):
    return f"média = {acc.mean():.3f} | pior cliente = {acc.min():.3f} | percentil 10 = {np.percentile(acc, 10):.3f}"


def ditto(treino, X, y, Wg, lam=0.5, epocas=5, eta=0.1, lote=32, seed=0):
    """Modelos pessoais v_k treinados com uma 'coleira' em direção ao modelo global Wg."""
    pessoais = []
    for k, p in enumerate(treino):
        v, rng = Wg.copy(), np.random.default_rng(seed + k)
        for _ in range(epocas):
            ordem = rng.permutation(len(p))
            for i in range(0, len(p), lote):
                b = p[ordem[i:i + lote]]
                v -= eta * gradiente(v, X[b], y[b], lam, Wg)
        pessoais.append(v)
    return pessoais


def particao_conceitos(y, n_clientes=12, grupos=3, seed=0):
    """Mudança de conceito: cada grupo de clientes usa uma troca diferente dos nomes das classes."""
    rng = np.random.default_rng(seed)
    perms = [np.arange(10)] + [rng.permutation(10) for _ in range(grupos - 1)]
    partes = particao_iid(y, n_clientes, seed)
    grupo = np.arange(n_clientes) % grupos
    y_novo = y.copy()
    for k, p in enumerate(partes):
        y_novo[p] = perms[grupo[k]][y[p]]
    return partes, y_novo, grupo


def ifca(treino, X, y, K=3, rodadas=30, epocas=5, eta=0.1, lote=32, seed=0):
    """Agrupamento iterativo: cada cliente escolhe o modelo de menor perda; o servidor faz a média por grupo."""
    rng = np.random.default_rng(seed)
    modelos = [rng.normal(0, 0.05, (X.shape[1], 10)) for _ in range(K)]
    for r in range(rodadas):
        locais, tamanhos = [[] for _ in range(K)], [[] for _ in range(K)]
        escolha = []
        for c, p in enumerate(treino):
            j = int(np.argmin([perda(W, X[p], y[p]) for W in modelos]))
            escolha.append(j)
            locais[j].append(treino_local(modelos[j], X[p], y[p], epocas, eta, lote, 0.0, seed * 1000 + r * 50 + c))
            tamanhos[j].append(len(p))
        for j in range(K):
            if locais[j]:
                pj = np.array(tamanhos[j]) / sum(tamanhos[j])
                modelos[j] = sum(a * b for a, b in zip(pj, locais[j]))
    return modelos, escolha
