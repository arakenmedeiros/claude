"""Capítulo 2: códigos de referência das questões discursivas de programação."""

# %% [Questão discursiva 14]
import numpy as np

def softmax(z):
    z = np.asarray(z, float)
    e = np.exp(z - z.max())
    return e / e.sum()

def log_softmax(z):
    z = np.asarray(z, float)
    m = z.max()
    return z - m - np.log(np.exp(z - m).sum())

z = np.array([2.0, 1.0, 0.1])
assert np.allclose(log_softmax(z), np.log(softmax(z)))
g = np.array([1000.0, 999.0, 998.0])
assert np.all(np.isfinite(log_softmax(g)))
print(log_softmax(z).round(4), log_softmax(g).round(4))

# %% [Questão discursiva 15]
import numpy as np

def log_softmax(Z):
    Z = np.asarray(Z, float)
    Z = Z - Z.max(axis=1, keepdims=True)
    return Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))

def entropia_cruzada_lote(logits, alvos):
    lp = log_softmax(logits)
    return float(-lp[np.arange(len(alvos)), alvos].mean())

# dois exemplos, três classes, logits todos iguais -> p = 1/3 -> perda = ln 3
L = np.zeros((2, 3))
assert abs(entropia_cruzada_lote(L, np.array([0, 2])) - np.log(3)) < 1e-12
print(entropia_cruzada_lote(L, np.array([0, 2])))

# %% [Questão discursiva 16]
def descida(x0, eta, tol=1e-6, max_passos=10_000):
    x = x0
    for i in range(max_passos):
        g = 2 * (x - 3)
        if abs(g) < tol:
            return x, i
        x = x - eta * g
        if abs(x) > 1e12:
            return float("inf"), i
    return x, max_passos

x, n = descida(0.0, 0.1)
assert abs(x - 3) < 1e-5
print("eta=0.1:", round(x, 6), n, "passos")
x2, n2 = descida(0.0, 1.1)
assert x2 == float("inf")
print("eta=1.1: diverge após", n2, "passos")

# %% [Questão discursiva 17]
def memoria_gb(n_parametros, bits):
    return n_parametros * bits / 8 / 1e9

def cabe(n_parametros, bits, gb_disponiveis, margem=0.2):
    return memoria_gb(n_parametros, bits) <= gb_disponiveis * (1 - margem)

assert abs(memoria_gb(7e9, 16) - 14.0) < 1e-9
assert abs(memoria_gb(7e9, 4) - 3.5) < 1e-9
print("16 bits cabe em 16 GB com 20% de margem?", cabe(7e9, 16, 16))
print("4 bits cabe em 16 GB com 20% de margem?", cabe(7e9, 4, 16))
