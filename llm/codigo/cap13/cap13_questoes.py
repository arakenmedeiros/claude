"""Capítulo 13: códigos de referência das questões discursivas de programação."""

# %% [Questão discursiva 11]
import numpy as np

def num_patches(h, l, p):
    assert h % p == 0 and l % p == 0
    return (h // p) * (l // p)

def patches_flat(img, p):
    h, l = img.shape
    return img.reshape(h // p, p, l // p, p).transpose(0, 2, 1, 3).reshape(-1, p * p)

def reconstruir(P, h, l, p):
    return P.reshape(h // p, l // p, p, p).transpose(0, 2, 1, 3).reshape(h, l)

img = np.arange(64, dtype=float).reshape(8, 8)
P = patches_flat(img, 4)
assert P.shape == (num_patches(8, 8, 4), 16)
assert np.array_equal(reconstruir(P, 8, 8, 4), img)
print(P.shape, num_patches(224, 224, 16), "reconstrução exata")

# %% [Questão discursiva 12]
import numpy as np

def perda_contrastiva(cos, tau):
    S = cos / tau
    n = len(S)
    def ce(M):
        M = M - M.max(axis=1, keepdims=True)
        logp = M - np.log(np.exp(M).sum(axis=1, keepdims=True))
        return -np.mean(np.diag(logp))
    return 0.5 * (ce(S) + ce(S.T))

cos = np.array([[0.8, 0.2], [0.1, 0.6]])
L = perda_contrastiva(cos, 0.5)
print(round(L, 4))
assert abs(L - 0.2920) < 1e-3
print("perda com tau=5:", round(perda_contrastiva(cos, 5.0), 4), "| ln 2 =", round(np.log(2), 4))

# %% [Questão discursiva 13]
import numpy as np

fs = 8000
t = np.arange(fs) / fs
x = np.sin(2 * np.pi * 500 * t) + 0.5 * np.sin(2 * np.pi * 1500 * t)
mag = np.abs(np.fft.rfft(x))
f = np.fft.rfftfreq(len(x), 1 / fs)
picos = f[np.argsort(mag)[-2:]]
print(sorted(picos.tolist()))
assert sorted(picos.tolist()) == [500.0, 1500.0]
print("resolução (Hz):", fs / len(x))

# %% [Questão discursiva 14]
import numpy as np

T = 1000
beta = np.linspace(1e-4, 0.02, T)
abar = np.cumprod(1 - beta)

def x_t(x0, t, rng):
    eps = rng.standard_normal(x0.shape)
    return np.sqrt(abar[t]) * x0 + np.sqrt(1 - abar[t]) * eps

rng = np.random.default_rng(0)
x0 = np.full(200_000, 3.0)
for t in (99, 499):
    x = x_t(x0, t, rng)
    print(t + 1, round(x.mean(), 3), round(np.sqrt(abar[t]) * 3.0, 3), round(x.var(), 3), round(1 - abar[t], 3))
    assert abs(x.mean() - np.sqrt(abar[t]) * 3.0) < 0.01
    assert abs(x.var() - (1 - abar[t])) < 0.01
