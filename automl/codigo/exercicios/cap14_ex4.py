# Capítulo 14 — solução de referência (EX 4)
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

def gera(n=6000, vies=0.6, p_min=0.3, seed=0, infl=0.0):
    r = np.random.default_rng(seed)
    g = (r.random(n) < p_min).astype(int); z = r.normal(0, 1, n)
    renda = np.exp(0.5 * z + 0.35 * r.normal(size=n) - 0.25 * g) * 3.0 * (1 + infl)
    emprego = np.clip(5 + 2 * z + r.normal(0, 2, n), 0, None)
    divida = np.clip(0.4 - 0.1 * z + r.normal(0, 0.1, n), 0, 1)
    cep = 0.8 * g + 0.6 * r.normal(size=n); score = z + r.normal(0, 1, n)
    y_real = (z > 0).astype(int); y_hist = ((z - vies * g + r.normal(0, 0.5, n)) > 0).astype(int)
    return np.c_[renda, emprego, divida, cep, score], g, y_hist, y_real

X0, g0, y0, _ = gera(5000, seed=1); campeao = HistGradientBoostingClassifier(random_state=0).fit(X0[:, [0, 1, 2, 4]], y0)
ing, par, difs = 0, 0, []
rng = np.random.default_rng(0)
for s in range(20):
    X, g, yh, yr = gera(4000, seed=100 + s, infl=0.5); X = X[:, [0, 1, 2, 4]]
    des = HistGradientBoostingClassifier(random_state=0).fit(X[:2000], yr[:2000])
    ok_c, ok_d = campeao.predict(X[2000:]) == yr[2000:], des.predict(X[2000:]) == yr[2000:]
    dif = ok_d.astype(int) - ok_c.astype(int); n = len(dif); ad = ok_d.mean()
    ep = np.sqrt(ad * (1 - ad) / n); difs.append(dif.mean())
    ing += dif.mean() > 2 * ep                                                       # regra ingênua
    lo = np.percentile([dif[rng.integers(0, n, n)].mean() for _ in range(1000)], 2.5)
    par += lo > 0.005                                                                 # regra pareada com melhoria mínima
print(f"Diferença média (desafiante - campeão) = {np.mean(difs):+.3f} | mínima = {min(difs):+.3f} | máxima = {max(difs):+.3f}")
print(f"Regra ingênua assumiu em {ing} de 20 repetições | regra pareada (δ = 0,005) assumiu em {par} de 20")
