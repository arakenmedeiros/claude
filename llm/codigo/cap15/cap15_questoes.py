"""Capítulo 15: códigos de referência das questões discursivas de programação."""

# %% [Questão discursiva 14]
import numpy as np

def quantizar_grupos(W, bits=4, grupo=64):
    qmax = 2 ** (bits - 1) - 1
    Wg = W.reshape(-1, grupo)
    s = np.abs(Wg).max(axis=1, keepdims=True) / qmax
    s = np.where(s == 0, 1.0, s)
    q = np.clip(np.round(Wg / s), -qmax, qmax)
    return (q * s).reshape(W.shape)

rng = np.random.default_rng(0)
W = rng.normal(0, 0.02, (128, 256))
W[:, ::50] *= 25
erros = {}
for g in (256, 64, 16):
    Wq = quantizar_grupos(W, 4, g)
    erros[g] = np.sqrt(np.mean((W - Wq) ** 2)) / np.sqrt(np.mean(W ** 2))
    print(g, round(float(erros[g]), 4))
assert erros[16] < erros[64] < erros[256]

# %% [Questão discursiva 15]
import numpy as np

def simular_mm1(lam, mu, n=100000, seed=0):
    rng = np.random.default_rng(seed)
    chegada = np.cumsum(rng.exponential(1 / lam, n))
    servico = rng.exponential(1 / mu, n)
    saida = np.empty(n)
    anterior = 0.0
    for i in range(n):
        inicio = max(chegada[i], anterior)
        anterior = inicio + servico[i]
        saida[i] = anterior
    lat = saida - chegada
    return lat[n // 20:].mean()

for lam in (0.5, 0.8):
    sim = simular_mm1(lam, 1.0)
    teo = 1 / (1.0 - lam)
    print(lam, round(float(sim), 2), round(teo, 2))
    assert abs(sim - teo) / teo < 0.15

# %% [Questão discursiva 16]
import numpy as np

def tokens_esperados(alfa, gama):
    return (1 - alfa ** (gama + 1)) / (1 - alfa)

def ganho(alfa, gama, c):
    return tokens_esperados(alfa, gama) / (gama * c + 1)

def melhor_gama(alfa, c, gamas=range(1, 33)):
    return max(gamas, key=lambda g: ganho(alfa, g, c))

for a in (0.5, 0.8, 0.95):
    g = melhor_gama(a, 0.05)
    print(a, g, round(ganho(a, g, 0.05), 2))
assert melhor_gama(0.5, 0.05) < melhor_gama(0.95, 0.05)

rng = np.random.default_rng(0)
a, g, n = 0.8, 4, 200000
aceitos = rng.random((n, g)) < a
k = np.where(aceitos.all(1), g, np.argmin(aceitos, 1))
mc = (k + 1).mean()
print(round(float(mc), 3), round(tokens_esperados(a, g), 3))
assert abs(mc - tokens_esperados(a, g)) < 0.02

# %% [Questão discursiva 17]
import numpy as np

def psi(ref, atual, categorias, piso=1e-4):
    ref, atual = np.asarray(ref), np.asarray(atual)
    e = np.array([max(np.mean(ref == c), piso) for c in categorias])
    a = np.array([max(np.mean(atual == c), piso) for c in categorias])
    return float(np.sum((a - e) * np.log(a / e)))

cats = ["a", "b", "c", "d"]
rng = np.random.default_rng(0)
ref = rng.choice(cats, 2000)
sem = rng.choice(cats, 2000)
com = rng.choice(cats, 2000, p=[0.5, 0.5 / 3, 0.5 / 3, 0.5 / 3])    # 'a' passa de 25% a 50% (o dobro)
print(round(psi(ref, sem, cats), 4), round(psi(ref, com, cats), 4))
assert psi(ref, sem, cats) < 0.02 < psi(ref, com, cats)
