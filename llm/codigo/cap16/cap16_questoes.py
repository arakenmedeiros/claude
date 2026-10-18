"""Capítulo 16: códigos de referência das questões discursivas de programação."""

# %% [Questão discursiva 5]
import math
def n_prop(p, e, z=1.96): return math.ceil(z**2 * p * (1 - p) / e**2)
n3, n5 = n_prop(0.9, 0.03), n_prop(0.9, 0.05)
assert n3 == 385 and n5 == 139
print(n3, n5, round(n3 / n5, 2))

# %% [Questão discursiva 6]
a = [1] * 60 + [0] * 20 + [1] * 15 + [0] * 5
b = [1] * 60 + [0] * 20 + [0] * 15 + [1] * 5
n = len(a)
po = sum(x == y for x, y in zip(a, b)) / n
pa, pb = sum(a) / n, sum(b) / n
pe = pa * pb + (1 - pa) * (1 - pb)
kappa = (po - pe) / (1 - pe)
from sklearn.metrics import cohen_kappa_score
assert abs(kappa - cohen_kappa_score(a, b)) < 1e-12
print(round(po, 3), round(pe, 3), round(kappa, 3))

# %% [Questão discursiva 8]
import numpy as np
from scipy.stats import chi2, norm
delta, psi = 0.04, 0.15
za, zb = norm.ppf(0.975), norm.ppf(0.80)
n = int(np.ceil((za * np.sqrt(psi) + zb * np.sqrt(psi - delta**2))**2 / delta**2))
rng = np.random.default_rng(0)
c = rng.multinomial(n, [(psi + delta) / 2, (psi - delta) / 2, 1 - psi], size=4000)
b, cc = c[:, 0], c[:, 1]
x2 = (np.abs(b - cc) - 1) ** 2 / np.maximum(b + cc, 1)
poder = np.mean(chi2.sf(x2, 1) < 0.05)
assert 0.70 < poder < 0.90
print(n, round(float(poder), 3))

# %% [Questão discursiva 10]
import numpy as np
from sklearn.metrics import f1_score

def f1_macro_micro(y_true, y_pred, classes):
    tp = {c: 0 for c in classes}; fp = {c: 0 for c in classes}; fn = {c: 0 for c in classes}
    for t, p in zip(y_true, y_pred):
        if t == p: tp[t] += 1
        else:
            if p in fp: fp[p] += 1
            if t in fn: fn[t] += 1
    def f1(a, b, c): return 2 * a / (2 * a + b + c) if (2 * a + b + c) else 0.0
    macro = np.mean([f1(tp[c], fp[c], fn[c]) for c in classes])
    micro = f1(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    return macro, micro

y_true = ["a", "a", "b", "b", "b", "c", "c", "a", "b", "c"]
y_pred = ["a", "b", "b", "b", "a", "c", "a", "a", "b", "c"]
ma, mi = f1_macro_micro(y_true, y_pred, ["a", "b", "c"])
assert abs(ma - f1_score(y_true, y_pred, average="macro")) < 1e-12
assert abs(mi - f1_score(y_true, y_pred, average="micro")) < 1e-12
print(round(ma, 3), round(mi, 3))

# %% [Questão discursiva 11]
from scipy.stats import binomtest

def mcnemar_exato(a_ok, b_ok):
    b = sum(1 for x, y in zip(a_ok, b_ok) if x and not y)
    c = sum(1 for x, y in zip(a_ok, b_ok) if y and not x)
    d = b + c
    if d == 0: return b, c, 1.0
    return b, c, min(1.0, binomtest(min(b, c), d, 0.5).pvalue)

a = [1] * 40 + [1] * 12 + [0] * 3 + [0] * 45
b = [1] * 40 + [0] * 12 + [1] * 3 + [0] * 45
bb, cc, p = mcnemar_exato(a, b)
assert (bb, cc) == (12, 3) and 0.0 < p < 0.05
print(bb, cc, round(p, 4))

# %% [Questão discursiva 12]
def lcs(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            dp[i + 1][j + 1] = dp[i][j] + 1 if a[i] == b[j] else max(dp[i][j + 1], dp[i + 1][j])
    return dp[-1][-1]

def rouge_l_f(ref, hip):
    r, h = ref.lower().split(), hip.lower().split()
    l = lcs(r, h)
    if l == 0: return 0.0
    p, rc = l / len(h), l / len(r)
    return 2 * p * rc / (p + rc)

ref = "a multa por atraso é de 1,50 por dia"
errada = "a multa por atraso é de 5,00 por dia"
correta_parafrase = "cada dia de atraso custa 1,50"
assert rouge_l_f(ref, errada) > rouge_l_f(ref, correta_parafrase)
print(round(rouge_l_f(ref, errada), 3), round(rouge_l_f(ref, correta_parafrase), 3))

# %% [Questão discursiva 13]
import numpy as np

def portao_regressao(base_ok, novo_ok, estratos, margem=0.03, queda_estrato=0.10, n=2000, seed=0):
    base_ok, novo_ok = np.asarray(base_ok, float), np.asarray(novo_ok, float)
    d = novo_ok - base_ok
    rng = np.random.default_rng(seed)
    medias = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    lo = np.quantile(medias, 0.025)
    motivos = []
    if lo < -margem: motivos.append("queda geral")
    est = np.asarray(estratos)
    for e in sorted(set(est)):
        m = est == e
        if d[m].mean() < -queda_estrato: motivos.append(f"queda no estrato {e}")
    return not motivos, motivos

rng = np.random.default_rng(1)
est = np.array(["x"] * 400 + ["y"] * 100)
base = (rng.random(500) < 0.9).astype(float)
novo = base.copy()
idx = np.where((est == "y") & (base == 1))[0][:15]
novo[idx] = 0
ok, mot = portao_regressao(base, novo, est)
assert not ok and any("y" in m for m in mot)
print(ok, mot)
