"""fedlib.py - biblioteca didática do livro 'Aprendizado Federado'.
Simula clientes e servidor em um único computador, usando NumPy e scikit-learn."""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def carregar_digits(seed=0):
    """Dígitos manuscritos da UCI (via scikit-learn), padronizados, com coluna de viés."""
    X, y = load_digits(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    esc = StandardScaler().fit(Xtr)
    mais_um = lambda A: np.hstack([esc.transform(A), np.ones((len(A), 1))])
    return mais_um(Xtr), mais_um(Xte), ytr, yte


def particao_iid(y, k, seed=0):
    """Embaralha tudo e reparte em k clientes de tamanho parecido."""
    idx = np.random.default_rng(seed).permutation(len(y))
    return [np.array(p) for p in np.array_split(idx, k)]


def particao_shards(y, k, por_cliente=2, seed=0):
    """Ordena por classe, corta em k*por_cliente pedaços e sorteia pedaços para cada cliente."""
    rng = np.random.default_rng(seed)
    pedacos = np.array_split(np.argsort(y, kind="stable"), k * por_cliente)
    ordem = rng.permutation(len(pedacos))
    return [np.concatenate([pedacos[ordem[i * por_cliente + j]] for j in range(por_cliente)]) for i in range(k)]


def particao_dirichlet(y, k, alfa=0.5, seed=0):
    """Cada classe é repartida entre os clientes com proporções sorteadas de uma Dirichlet(alfa)."""
    rng = np.random.default_rng(seed)
    partes = [[] for _ in range(k)]
    for c in np.unique(y):
        ic = rng.permutation(np.where(y == c)[0])
        p = rng.dirichlet([alfa] * k)
        cortes = (np.cumsum(p) * len(ic)).astype(int)[:-1]
        for i, pedaco in enumerate(np.split(ic, cortes)):
            partes[i] += pedaco.tolist()
    return [np.array(p, dtype=int) for p in partes]


def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def gradiente(W, X, y, mu=0.0, W_ref=None):
    """Gradiente da entropia cruzada (regressão logística multiclasse) + termo proximal opcional."""
    P = softmax(X @ W)
    P[np.arange(len(y)), y] -= 1.0
    g = X.T @ P / len(y)
    if mu > 0 and W_ref is not None:
        g = g + mu * (W - W_ref)
    return g


def treino_local(W, X, y, epocas=1, eta=0.1, lote=32, mu=0.0, seed=0):
    """Treino local do cliente: mini-batch SGD a partir do modelo global W."""
    rng = np.random.default_rng(seed)
    W0, W = W, W.copy()
    for _ in range(epocas):
        ordem = rng.permutation(len(y))
        for i in range(0, len(y), lote):
            b = ordem[i:i + lote]
            W -= eta * gradiente(W, X[b], y[b], mu, W0)
    return W


def acuracia(W, X, y):
    return float(np.mean(np.argmax(X @ W, axis=1) == y))


def perda(W, X, y):
    P = softmax(X @ W)
    return float(-np.mean(np.log(P[np.arange(len(y)), y] + 1e-12)))


def fedavg(partes, X, y, Xte, yte, rodadas=30, epocas=1, eta=0.1, lote=32,
           frac=1.0, mu=0.0, seed=0, W0=None, classes=10):
    """FedAvg: em cada rodada, clientes sorteados treinam localmente e o servidor faz a média ponderada."""
    rng = np.random.default_rng(seed)
    W = np.zeros((X.shape[1], classes)) if W0 is None else W0.copy()
    historico = []
    for r in range(rodadas):
        k = max(1, int(round(frac * len(partes))))
        escolhidos = rng.choice(len(partes), k, replace=False)
        locais, tamanhos = [], []
        for c in escolhidos:
            if len(partes[c]) == 0:
                continue
            locais.append(treino_local(W, X[partes[c]], y[partes[c]], epocas, eta, lote, mu, seed * 1000 + r * 50 + int(c)))
            tamanhos.append(len(partes[c]))
        p = np.array(tamanhos) / sum(tamanhos)
        W = sum(pi * Wi for pi, Wi in zip(p, locais))
        historico.append(acuracia(W, Xte, yte))
    return W, historico
