# Capítulo 13 — solução de referência (PROJ)
import warnings; warnings.filterwarnings("ignore")
import time
import numpy as np
from flaml import AutoML
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def gera(n=6000, vies=0.6, p_min=0.3, seed=0):
    r = np.random.default_rng(seed)
    g = (r.random(n) < p_min).astype(int); z = r.normal(0, 1, n)
    renda = np.exp(0.5 * z + 0.35 * r.normal(size=n) - 0.25 * g) * 3.0
    emprego = np.clip(5 + 2 * z + r.normal(0, 2, n), 0, None)
    divida = np.clip(0.4 - 0.1 * z + r.normal(0, 0.1, n), 0, 1)
    cep = 0.8 * g + 0.6 * r.normal(size=n); score = z + r.normal(0, 1, n)
    y_real = (z > 0).astype(int); y_hist = ((z - vies * g + r.normal(0, 0.5, n)) > 0).astype(int)
    return np.c_[renda, emprego, divida, cep, score], g, y_hist, y_real

NOMES = ["renda", "tempo_emprego", "razao_divida", "cep_score", "score_historico"]
X, g, yh, yr = gera(vies=0.4, p_min=0.2, seed=7)
Xa, Xte, ga, gte, ya, yte, yra, yrte = train_test_split(X, g, yh, yr, test_size=0.3, random_state=0, stratify=yh)
Xtr, Xva, gtr, gva, ytr, yva, yrtr, yrva = train_test_split(Xa, ga, ya, yra, test_size=0.25, random_state=1, stratify=ya)
t0 = time.time(); a = AutoML()
a.fit(Xtr, ytr, task="classification", time_budget=120, max_iter=20, metric="accuracy", eval_method="cv", n_splits=3, seed=0, verbose=0)
tempo = time.time() - t0
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr)
print(f"FLAML ({a.best_estimator}): acc = {(a.predict(Xte) == yte).mean():.3f} | Reg. logística: acc = {lr.score(Xte, yte):.3f}")
pi = permutation_importance(a.model.estimator, Xte, yte, n_repeats=10, random_state=0)
print("Permutação (FLAML):", {n: round(float(v), 3) for n, v in zip(NOMES, pi.importances_mean)})

def rel(nome, pred):
    s = [pred[gte == k].mean() for k in (0, 1)]; t = [pred[(gte == k) & (yrte == 1)].mean() for k in (0, 1)]
    print(f"{nome:<22s} acc hist = {(pred == yte).mean():.3f} | acc verdade = {(pred == yrte).mean():.3f} | seleção {s[0]:.2f}/{s[1]:.2f} | TPR real {t[0]:.2f}/{t[1]:.2f}")
rel("FLAML", a.predict(Xte)); rel("Reg. logística", lr.predict(Xte))
pg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, gtr)
print("AUC para prever o grupo pelos atributos: %.3f" % roc_auc_score(gte, pg.predict_proba(Xte)[:, 1]))
pva, p = a.predict_proba(Xva)[:, 1], a.predict_proba(Xte)[:, 1]
sel0 = (pva[gva == 0] >= 0.5).mean(); t1 = np.quantile(pva[gva == 1], 1 - sel0)      # limiar escolhido na VALIDAÇÃO
rel("FLAML + limiar (valid.)", np.where(gte == 0, p >= 0.5, p >= t1).astype(int))
print(f"Limiar do grupo 1 = {t1:.2f} | melhorias em config_history = {len(a.config_history)} | tempo de busca = {tempo:.0f} s")
