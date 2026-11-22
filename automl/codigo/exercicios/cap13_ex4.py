# Capítulo 13 — solução de referência (EX 4)
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

def gera(n=6000, vies=0.6, p_min=0.3, seed=0):
    r = np.random.default_rng(seed)
    g = (r.random(n) < p_min).astype(int); z = r.normal(0, 1, n)
    renda = np.exp(0.5 * z + 0.35 * r.normal(size=n) - 0.25 * g) * 3.0
    emprego = np.clip(5 + 2 * z + r.normal(0, 2, n), 0, None)
    divida = np.clip(0.4 - 0.1 * z + r.normal(0, 0.1, n), 0, 1)
    cep = 0.8 * g + 0.6 * r.normal(size=n); score = z + r.normal(0, 1, n)
    y_real = (z > 0).astype(int); y_hist = ((z - vies * g + r.normal(0, 0.5, n)) > 0).astype(int)
    return np.c_[renda, emprego, divida, cep, score], g, y_hist, y_real

X, g, yh, yr = gera()
Xa, Xte, ga, gte, ya, yte, yra, yrte = train_test_split(X, g, yh, yr, test_size=0.3, random_state=0, stratify=yh)
Xtr, Xva, gtr, gva, ytr, yva, yrtr, yrva = train_test_split(Xa, ga, ya, yra, test_size=0.25, random_state=1, stratify=ya)
modelo = HistGradientBoostingClassifier(random_state=0).fit(Xtr, ytr)
pva, p = modelo.predict_proba(Xva)[:, 1], modelo.predict_proba(Xte)[:, 1]
alvo = (pva[(gva == 0) & (yva == 1)] >= 0.5).mean()                 # TPR histórico do grupo 0, na VALIDAÇÃO
cand = np.linspace(0.05, 0.9, 171)
t1 = min(cand, key=lambda t: abs((pva[(gva == 1) & (yva == 1)] >= t).mean() - alvo))   # limiar escolhido na validação
for nome, pred in (("limiar único 0,5", (p >= 0.5).astype(int)), (f"limiar do grupo 1 = {t1:.2f}", np.where(gte == 0, p >= 0.5, p >= t1).astype(int))):
    tprh = [pred[(gte == k) & (yte == 1)].mean() for k in (0, 1)]; tprr = [pred[(gte == k) & (yrte == 1)].mean() for k in (0, 1)]
    print(f"{nome:<24s} TPR histórico = {tprh[0]:.2f}/{tprh[1]:.2f} | TPR real = {tprr[0]:.2f}/{tprr[1]:.2f} | acc verdade = {(pred == yrte).mean():.3f}")
