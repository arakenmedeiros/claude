"""Capítulo 17: códigos de referência das questões discursivas de programação."""

# %% [Questão discursiva 7]
import numpy as np

y_true = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0])
y_pred = np.array([1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0])
g      = np.array(["A"] * 6 + ["B"] * 6)

def dif_paridade(y_pred, g):
    return y_pred[g == "A"].mean() - y_pred[g == "B"].mean()

def dif_oportunidade(y_true, y_pred, g):
    a = (g == "A") & (y_true == 1); b = (g == "B") & (y_true == 1)
    return y_pred[a].mean() - y_pred[b].mean()

dp, io = dif_paridade(y_pred, g), dif_oportunidade(y_true, y_pred, g)
assert abs(dp - 1 / 3) < 1e-9 and abs(io) < 1e-9, (dp, io)
print(f"paridade demográfica: {dp:.4f} | igualdade de oportunidades: {io:.4f}")

# %% [Questão discursiva 11]
import re

def cpf_valido(texto):
    d = re.sub(r"\D", "", texto)
    if len(d) != 11 or len(set(d)) == 1:
        return False
    for k in (9, 10):
        s = sum(int(d[i]) * (k + 1 - i) for i in range(k))
        r = (s * 10) % 11
        if (0 if r == 10 else r) != int(d[k]):
            return False
    return True

assert cpf_valido("529.982.247-25") and cpf_valido("52998224725")
assert not cpf_valido("111.111.111-11") and not cpf_valido("529.982.247-24") and not cpf_valido("123")
print("ok")

# %% [Questão discursiva 12]
def destino_permitido(destino, dominios, subdominios=False):
    if destino.count("@") != 1:
        return False
    dom = destino.split("@")[1].lower()
    for d in dominios:
        if dom == d or (subdominios and dom.endswith("." + d)):
            return True
    return False

P = ["aurora.example"]
assert destino_permitido("a@aurora.example", P)
assert not destino_permitido("a@aurora.example.coleta.example", P)
assert not destino_permitido("a@outro.example", P)
assert destino_permitido("a@sub.aurora.example", P, subdominios=True) and not destino_permitido("a@sub.aurora.example", P)
assert not destino_permitido("a@b@aurora.example", P)
print("ok")

# %% [Questão discursiva 13]
import numpy as np

def wilson(k, n, z=1.96):
    p = k / n; c = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / c
    meia = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / c
    return centro - meia, centro + meia

def normal(k, n, z=1.96):
    p = k / n; e = z * np.sqrt(p * (1 - p) / n)
    return p - e, p + e

lo_w, hi_w = wilson(3, 40); lo_n, hi_n = normal(3, 40)
print(f"Wilson [{lo_w:.3f}; {hi_w:.3f}] | normal [{lo_n:.3f}; {hi_n:.3f}]")
assert lo_n < 0 <= lo_w and 0.02 < lo_w < 0.03 and 0.19 < hi_w < 0.20

# %% [Questão discursiva 14]
import numpy as np
from collections import Counter

def k_min(regs):
    return min(Counter(regs).values())

def k_anonimato(regs, generalizar):
    return k_min(regs), k_min([generalizar(r) for r in regs])

rng = np.random.default_rng(0)
regs = [(int(c), int(i)) for c, i in zip(rng.integers(60000, 60040, 600), rng.integers(18, 80, 600))]
antes, depois = k_anonimato(regs, lambda r: (r[0] // 100, (r[1] // 10) * 10))
print("k antes:", antes, "| k depois:", depois)
assert antes == 1 and depois > antes
