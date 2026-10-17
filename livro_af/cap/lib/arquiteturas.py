import numpy as np


def sigmoide(z):
    return 1 / (1 + np.exp(-np.clip(z, -50, 50)))


# ---------------------------------------------------------------- Aprendizado federado vertical
def vfl_treinar(X_partes, y, epocas=300, eta=0.5):
    """Regressão logística vertical: cada parte tem só as suas colunas; apenas a parte 0 conhece o rótulo y.
    Cada parte envia z_k = X_k w_k; a parte 0 soma, calcula o resíduo (p - y) e o devolve."""
    n = len(y)
    w = [np.zeros(X.shape[1]) for X in X_partes]
    b = 0.0
    historico = []
    for _ in range(epocas):
        z = sum(X @ wk for X, wk in zip(X_partes, w)) + b          # soma das contribuições parciais
        p = sigmoide(z)
        residuo = p - y                                             # calculado pela parte com o rótulo
        for k, X in enumerate(X_partes):
            w[k] -= eta * X.T @ residuo / n                         # cada parte atualiza só os seus pesos
        b -= eta * residuo.mean()
        historico.append(-np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)))
    return w, b, historico


def vfl_prever(X_partes, w, b):
    return (sum(X @ wk for X, wk in zip(X_partes, w)) + b > 0).astype(int)


# ---------------------------------------------------------------- Split learning
def treinar_junto(X, y, W1, W2, eta=0.1, passos=50):
    """Rede de 2 camadas (tanh + softmax) treinada de forma convencional."""
    W1, W2 = W1.copy(), W2.copy()
    Y = np.eye(W2.shape[1])[y]
    for _ in range(passos):
        H = np.tanh(X @ W1)
        P = np.exp(H @ W2); P /= P.sum(1, keepdims=True)
        dZ = (P - Y) / len(y)
        dW2 = H.T @ dZ
        dH = dZ @ W2.T * (1 - H ** 2)
        W2 -= eta * dW2
        W1 -= eta * X.T @ dH
    return W1, W2


def treinar_dividido(X, y, W1, W2, eta=0.1, passos=50):
    """Mesma rede, mas dividida: o cliente tem W1 e os dados; o servidor tem W2 e os rótulos.
    Só trafegam as ativações H ('smashed data') e o gradiente em H."""
    W1, W2 = W1.copy(), W2.copy()
    Y = np.eye(W2.shape[1])[y]
    for _ in range(passos):
        H = np.tanh(X @ W1)                                         # cliente: passagem direta até a camada de corte
        # --- H sobe ao servidor ---
        P = np.exp(H @ W2); P /= P.sum(1, keepdims=True)           # servidor: conclui a passagem direta
        dZ = (P - Y) / len(y)
        dH = dZ @ W2.T                                              # servidor: gradiente na camada de corte
        W2 -= eta * H.T @ dZ
        # --- dH desce ao cliente ---
        W1 -= eta * X.T @ (dH * (1 - H ** 2))                       # cliente: conclui a retropropagação
    return W1, W2


# ---------------------------------------------------------------- Aprendizado descentralizado
def matriz_mistura(topologia, n):
    """Matriz duplamente estocástica de mistura para uma topologia de comunicação."""
    A = np.zeros((n, n))
    if topologia == "anel":
        for i in range(n):
            A[i, i] = 1 / 3; A[i, (i + 1) % n] = 1 / 3; A[i, (i - 1) % n] = 1 / 3
    elif topologia == "completa":
        A[:] = 1 / n
    elif topologia == "estrela":
        for i in range(1, n):
            A[0, i] = A[i, 0] = 1 / n
            A[i, i] = 1 - 1 / n
        A[0, 0] = 1 - (n - 1) / n
    return A


def consenso(valores, A, rodadas):
    """Cada nó repete: novo valor = média ponderada dos valores dos vizinhos. Devolve o erro até a média global."""
    x = valores.copy()
    alvo = x.mean(axis=0)
    erros = [np.abs(x - alvo).max()]
    for _ in range(rodadas):
        x = A @ x
        erros.append(np.abs(x - alvo).max())
    return erros
