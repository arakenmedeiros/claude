"""Funções de ativação e suas derivadas (implementação didática do livro).

Convenção: cada ativação é um par (f, df), onde df(z) devolve phi'(z)
calculada a partir da entrada líquida z (pré-ativação).
"""
import numpy as np


def sigmoide(z):
    """Logística 1/(1+e^-z), estável para |z| grande."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def d_sigmoide(z):
    s = sigmoide(z)
    return s * (1.0 - s)


def tanh(z):
    return np.tanh(z)


def d_tanh(z):
    return 1.0 - np.tanh(z) ** 2


def relu(z):
    return np.maximum(0.0, z)


def d_relu(z):
    # Convenção: derivada 0 em z = 0 (subgradiente).
    return (np.asarray(z) > 0).astype(float)


def identidade(z):
    return np.asarray(z, dtype=float)


def d_identidade(z):
    return np.ones_like(np.asarray(z, dtype=float))


def softmax(z):
    """Softmax por linhas (amostras nas linhas), estável."""
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def d_sigmoide_fahlman(z):
    """Derivada da sigmoide + 0,1 (artifício de Fahlman para evitar 'pontos planos').
    ATENÇÃO: NÃO é a derivada verdadeira; o gradiente resultante falha na verificação por diferenças finitas."""
    return d_sigmoide(z) + 0.1


ATIVACOES = {
    "sigmoide_fahlman": (sigmoide, d_sigmoide_fahlman),
    "sigmoide": (sigmoide, d_sigmoide),
    "tanh": (tanh, d_tanh),
    "relu": (relu, d_relu),
    "identidade": (identidade, d_identidade),
}
