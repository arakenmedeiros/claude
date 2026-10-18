"""Capítulo 5: códigos de referência das questões discursivas de programação."""

# %% [Questão discursiva 15]
import numpy as np

def pares_causais(tokens):
    return [(tokens[:t], tokens[t]) for t in range(1, len(tokens))]

def mascarar(tokens, p=0.15, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.random(len(tokens)) < p
    entrada = ["[MASK]" if x else t for t, x in zip(tokens, m)]
    alvos = [t for t, x in zip(tokens, m) if x]
    return entrada, alvos

f = "o aluno lê o livro .".split()
pc = pares_causais(f)
assert len(pc) == len(f) - 1 and pc[0] == (["o"], "aluno") and pc[-1][1] == "."
ent, alv = mascarar(f, p=0.5, seed=1)
assert len(ent) == len(f) and ent.count("[MASK]") == len(alv)
print(pc[1], ent, alv)

# %% [Questão discursiva 16]
import numpy as np

def taxa(s, S, eta_max=1e-3, eta_min=1e-4, w=100):
    if s <= w:
        return eta_max * s / w
    return eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * (s - w) / (S - w)))

assert abs(taxa(50, 1000) - 5e-4) < 1e-12
assert abs(taxa(550, 1000) - 5.5e-4) < 1e-12
assert abs(taxa(100, 1000) - 1e-3) < 1e-12
assert abs(taxa(1000, 1000) - 1e-4) < 1e-12
print(taxa(50, 1000), taxa(550, 1000))

# %% [Questão discursiva 17]
def shingles(texto, k=3):
    p = texto.lower().split()
    return {tuple(p[i:i + k]) for i in range(len(p) - k + 1)}

def jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else 0.0

def deduplicar(docs, k=3, limiar=0.5):
    S = [shingles(d, k) for d in docs]
    manter = []
    for i, s in enumerate(S):
        if all(jaccard(s, S[j]) < limiar for j in manter):
            manter.append(i)
    return manter

docs = [
    "a biblioteca central funciona de segunda a sexta das sete às vinte e duas horas",
    "a biblioteca central funciona de segunda a sexta das sete às vinte e duas horas",
    "a biblioteca central funciona de segunda a sexta das sete às vinte e três horas",
    "o restaurante universitário serve almoço e jantar todos os dias da semana letiva",
]
assert deduplicar(docs, k=3, limiar=0.5) == [0, 3]
print(deduplicar(docs, k=3, limiar=0.5))
