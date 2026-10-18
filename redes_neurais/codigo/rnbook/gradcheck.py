"""Verificação de gradientes por diferenças finitas (implementação didática).

Limitações: só verifica diferenciabilidade local no ponto testado; erros
de truncamento O(h^2) e de arredondamento O(eps/h) competem; ReLU em z=0
e outras não diferenciabilidades quebram a comparação; não prova que o
gradiente está correto em outros pontos.
"""
import numpy as np


def grad_numerico(f, theta, h=1e-6):
    """Diferença central: (f(theta+h e_i) - f(theta-h e_i)) / (2h)."""
    g = np.zeros_like(theta)
    for i in range(theta.size):
        d = np.zeros_like(theta)
        d[i] = h
        g[i] = (f(theta + d) - f(theta - d)) / (2.0 * h)
    return g


def erro_relativo(g_analitico, g_numerico, eps=1e-12):
    num = np.linalg.norm(g_analitico - g_numerico)
    den = np.linalg.norm(g_analitico) + np.linalg.norm(g_numerico) + eps
    return num / den
