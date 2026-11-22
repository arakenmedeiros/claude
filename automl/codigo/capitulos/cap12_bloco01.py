# Capítulo 12 — AutoML para séries temporais, texto e imagens
# Bloco 1. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, KFold

bruta = sm.datasets.co2.load_pandas().data["co2"]                     # série semanal, com semanas ausentes
serie = bruta.ffill().resample("MS").mean()                         # preenchimento CAUSAL: usa só o último valor já observado
y = serie.values; n = len(y); H = 36
print("Meses na série:", n, "| período:", serie.index[0].strftime("%m/%Y"), "a", serie.index[-1].strftime("%m/%Y"))

def monta(y, L, dif):
    X, alvo, base = [], [], []
    for t in range(max(L, 12), len(y)):
        b = y[t - 12] if dif else 0.0                     # com dif=True, prevemos a diferença para 12 meses antes
        X.append(np.r_[y[t - L:t][::-1] - b, (np.arange(12) == t % 12)]); alvo.append(y[t] - b); base.append(b)
    return np.array(X), np.array(alvo), np.array(base)

MODELOS = {"ridge": lambda: Ridge(1.0), "hgb": lambda: HistGradientBoostingRegressor(max_iter=100, random_state=0)}

def mae_cv(cfg, cv, ate):
    mod, L, dif = cfg; X, a, b = monta(y[:ate], L, dif); e = []
    for tr, va in cv.split(X):
        m = MODELOS[mod]().fit(X[tr], a[tr]); e.append(np.abs(m.predict(X[va]) - a[va]).mean())
    return np.mean(e)

for cfg in (("ridge", 12, True), ("hgb", 12, False)):
    a = mae_cv(cfg, KFold(5, shuffle=True, random_state=0), n - H)
    t = mae_cv(cfg, TimeSeriesSplit(5, test_size=12), n - H)
    print(f"{cfg[0]:<5s}: MAE com validação aleatória = {a:.3f} | com validação temporal = {t:.3f}")
