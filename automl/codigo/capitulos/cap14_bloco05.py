# Capítulo 14 — MLOps, implantação e projeto integrador
# Bloco 5. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
import warnings; warnings.filterwarnings("ignore")
import json
import numpy as np
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier
import os; os.makedirs("resultados", exist_ok=True)

def gera(n, seed, infl=0.0):
    r = np.random.default_rng(seed)
    g = (r.random(n) < 0.3).astype(int); z = r.normal(0, 1, n)
    renda = np.exp(0.5 * z + 0.35 * r.normal(size=n) - 0.25 * g) * 3.0 * (1 + infl)       # a renda sobe com a inflação
    emprego = np.clip(5 + 2 * z + r.normal(0, 2, n), 0, None); divida = np.clip(0.4 - 0.1 * z + r.normal(0, 0.1, n), 0, 1)
    score = z + r.normal(0, 1, n)
    return np.c_[renda, emprego, divida, score], (z > 0).astype(int)

def psi(ref, novo, k=10):
    cortes = np.quantile(ref, np.linspace(0, 1, k + 1)[1:-1])
    a = np.bincount(np.searchsorted(cortes, ref), minlength=k) / len(ref)
    b = np.bincount(np.searchsorted(cortes, novo), minlength=k) / len(novo)
    a, b = np.clip(a, 1e-4, None), np.clip(b, 1e-4, None)
    return float(np.sum((b - a) * np.log(b / a)))

Xtr, ytr = gera(5000, seed=1)
modelo = HistGradientBoostingClassifier(random_state=0).fit(Xtr, ytr)
print(f"{'mês':>3s} {'inflação':>9s} {'acurácia':>9s} {'KS (renda)':>11s} {'valor-p':>9s} {'PSI renda':>10s} {'PSI emprego':>12s}")
linhas = []
for mes in range(0, 7):
    infl = 0.10 * mes
    Xn, yn = gera(2000, seed=100 + mes, infl=infl)
    acc = modelo.score(Xn, yn); ks = ks_2samp(Xtr[:, 0], Xn[:, 0]); pr = psi(Xtr[:, 0], Xn[:, 0]); pe = psi(Xtr[:, 1], Xn[:, 1])
    linhas.append({"mes": mes, "acc": acc, "ks": float(ks.statistic), "psi_renda": pr, "psi_emp": pe})
    print(f"{mes:>3d} {infl:>8.0%} {acc:>9.3f} {ks.statistic:>11.3f} {ks.pvalue:>9.1e} {pr:>10.3f} {pe:>12.3f}")
json.dump(linhas, open("resultados/f14_deriva.json", "w"))
