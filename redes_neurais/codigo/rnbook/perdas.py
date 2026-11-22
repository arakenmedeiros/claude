"""Perdas numericamente estáveis (Capítulo 9): log-sum-exp, softmax, entropia cruzada a partir de logits.

Implementações didáticas do livro. Convenção: Z (N, K) são logits; T é one-hot (N, K) ou rótulos 0/1."""
import numpy as np


def logsumexp(Z, eixo=-1):
    """log sum exp(z) = m + log sum exp(z - m), com m = max(z): evita overflow."""
    m = np.max(Z, axis=eixo, keepdims=True)
    return (m + np.log(np.sum(np.exp(Z - m), axis=eixo, keepdims=True))).squeeze(eixo)


def softmax_estavel(Z):
    return np.exp(Z - logsumexp(Z)[..., None])


def softmax_ingenua(Z):
    E = np.exp(Z)
    return E / E.sum(axis=-1, keepdims=True)


def ce_logits(Z, T):
    """Entropia cruzada média: -sum_k t_k log softmax(z)_k = logsumexp(z) - sum_k t_k z_k. Gradiente: (softmax - T)/N."""
    N = Z.shape[0]
    perda = float(np.mean(logsumexp(Z) - np.sum(T * Z, axis=1)))
    grad = (softmax_estavel(Z) - T) / N
    return perda, grad


def ce_ingenua(Z, T):
    P = softmax_ingenua(Z)
    return float(-np.mean(np.sum(T * np.log(P), axis=1)))


def bce_logits(Z, T):
    """Entropia cruzada binária por rótulo (multirrótulo), média sobre amostras e SOMA sobre rótulos:
    max(z,0) - z t + log(1 + exp(-|z|)).  Gradiente em z: (sigmoide(z) - t)/N."""
    N = Z.shape[0]
    L = np.maximum(Z, 0) - Z * T + np.log1p(np.exp(-np.abs(Z)))
    sig = np.where(Z >= 0, 1 / (1 + np.exp(-np.abs(Z))), np.exp(-np.abs(Z)) / (1 + np.exp(-np.abs(Z))))
    return float(np.mean(np.sum(L, axis=1))), (sig - T) / N


def bce_ingenua(Z, T):
    P = 1 / (1 + np.exp(-Z))
    return float(np.mean(np.sum(-(T * np.log(P) + (1 - T) * np.log(1 - P)), axis=1)))
