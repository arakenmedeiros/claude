"""Backpropagation 'com laços' (escalar), para uma amostra por vez.

Implementação didática do livro. Serve para ver cada derivada parcial
separadamente e para validar a versão vetorizada (rede.MLP.grad).
Perda por amostra: E_p = 1/2 * sum_k (y_k - t_k)^2.
"""
import numpy as np
from .ativacoes import ATIVACOES


def backprop_escalar(W, b, nomes_ativ, x, t):
    """W: lista de matrizes (n_l, n_{l-1}); b: lista de vetores; x: (n0,); t: (m,).
    Retorna (E, dW, db) calculados com laços explícitos."""
    L = len(W)
    phi = [ATIVACOES[a][0] for a in nomes_ativ]
    dphi = [ATIVACOES[a][1] for a in nomes_ativ]
    # --- passo direto
    a = [np.asarray(x, float)]
    z = []
    for l in range(L):
        zl = np.zeros(W[l].shape[0])
        for j in range(W[l].shape[0]):
            s = b[l][j]
            for i in range(W[l].shape[1]):
                s += W[l][j, i] * a[l][i]
            zl[j] = s
        z.append(zl)
        a.append(phi[l](zl))
    E = 0.5 * sum((a[-1][k] - t[k]) ** 2 for k in range(len(t)))
    # --- passo reverso
    delta = [None] * L
    delta[L - 1] = np.array([(a[L][k] - t[k]) * dphi[L - 1](z[L - 1][k:k + 1])[0]
                             for k in range(len(t))])
    for l in range(L - 2, -1, -1):
        d = np.zeros(W[l].shape[0])
        for j in range(W[l].shape[0]):
            soma = 0.0
            for k in range(W[l + 1].shape[0]):
                soma += delta[l + 1][k] * W[l + 1][k, j]
            d[j] = soma * dphi[l](z[l][j:j + 1])[0]
        delta[l] = d
    dW = [np.zeros_like(w) for w in W]
    db = [np.zeros_like(v) for v in b]
    for l in range(L):
        for j in range(W[l].shape[0]):
            db[l][j] = delta[l][j]
            for i in range(W[l].shape[1]):
                dW[l][j, i] = delta[l][j] * a[l][i]
    return E, dW, db
