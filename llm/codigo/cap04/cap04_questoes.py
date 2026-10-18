"""Capítulo 4: códigos de referência das questões discursivas de programação."""

# %% [Questão discursiva 14]
import numpy as np

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def atencao(Q, K, V, mascara=None):
    S = Q @ K.T / np.sqrt(Q.shape[-1])
    if mascara is not None:
        S = np.where(mascara, S, -1e9)
    A = softmax(S)
    return A @ V, A

Q = K = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
V = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 3.0]])
out, A = atencao(Q, K, V)
assert np.allclose(A.sum(axis=1), 1)
assert np.allclose(out[0], [1.604, 1.599], atol=1e-3)
out_c, A_c = atencao(Q, K, V, np.tril(np.ones((3, 3), dtype=bool)))
assert np.allclose(A_c[0], [1, 0, 0])
print(out.round(3)[0], A_c.round(3)[1])

# %% [Questão discursiva 15]
import numpy as np

def separa(M, h):
    T, d = M.shape
    return M.reshape(T, h, d // h).transpose(1, 0, 2)

def junta(Mh):
    h, T, dh = Mh.shape
    return Mh.transpose(1, 0, 2).reshape(T, h * dh)

M = np.arange(6 * 8, dtype=float).reshape(6, 8)
Mh = separa(M, 4)
assert Mh.shape == (4, 6, 2)
assert np.array_equal(junta(Mh), M)
assert np.array_equal(Mh[1, 0], M[0, 2:4])      # a cabeça 1 do token 0 contém as dimensões 2 e 3
print(Mh.shape)

# %% [Questão discursiva 16]
import numpy as np

def teste_vazamento(modelo, X, k, seed=0):
    rng = np.random.default_rng(seed)
    X2 = X.copy()
    X2[k:] = rng.normal(size=X2[k:].shape)
    return bool(np.allclose(modelo(X)[:k], modelo(X2)[:k]))

causal = lambda X: np.cumsum(X, axis=0)
nao_causal = lambda X: np.broadcast_to(X.mean(axis=0), X.shape).copy()

X = np.random.default_rng(1).normal(size=(8, 4))
assert teste_vazamento(causal, X, 5) is True
assert teste_vazamento(nao_causal, X, 5) is False
print(teste_vazamento(causal, X, 5), teste_vazamento(nao_causal, X, 5))

# %% [Questão discursiva 17]
import numpy as np

def posicional(T, d):
    pos = np.arange(T)[:, None]
    i = np.arange(d // 2)[None, :]
    ang = pos / (10000 ** (2 * i / d))
    PE = np.zeros((T, d))
    PE[:, 0::2] = np.sin(ang)
    PE[:, 1::2] = np.cos(ang)
    return PE

PE = posicional(10, 4)
assert PE.shape == (10, 4)
assert np.allclose(PE[0], [0, 1, 0, 1])
assert np.allclose(PE[1], [np.sin(1), np.cos(1), np.sin(0.01), np.cos(0.01)])
print(PE[1].round(4))
