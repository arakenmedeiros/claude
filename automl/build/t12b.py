import warnings; warnings.filterwarnings("ignore")
import itertools, json
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, KFold

serie = sm.datasets.co2.load_pandas().data["co2"].interpolate().resample("MS").mean()
y = serie.values; n = len(y); H = 36
def monta(y, L, dif):
    X, alvo, base = [], [], []
    for t in range(max(L, 12), len(y)):
        b = y[t - 12] if dif else 0.0
        X.append(np.r_[y[t - L:t][::-1] - b, (np.arange(12) == t % 12)]); alvo.append(y[t] - b); base.append(b)
    return np.array(X), np.array(alvo), np.array(base)
MODELOS = {"ridge": lambda: Ridge(1.0), "hgb": lambda: HistGradientBoostingRegressor(max_iter=100, random_state=0)}
def mae_cv(cfg, cv, ate):
    mod, L, dif = cfg; X, a, b = monta(y[:ate], L, dif); e = []
    for tr, va in cv.split(X):
        m = MODELOS[mod]().fit(X[tr], a[tr]); e.append(np.abs(m.predict(X[va]) + b[va] - (a[va] + b[va])).mean())
    return np.mean(e)
cfg = ("ridge", 12, True)
print("ridge: MAE com validação ALEATÓRIA   :", round(mae_cv(cfg, KFold(5, shuffle=True, random_state=0), n - H), 3))
print("ridge: MAE com validação TEMPORAL    :", round(mae_cv(cfg, TimeSeriesSplit(5, test_size=12), n - H), 3))
cfg = ("hgb", 12, False)
print("hgb  : MAE com validação ALEATÓRIA   :", round(mae_cv(cfg, KFold(5, shuffle=True, random_state=0), n - H), 3))
print("hgb  : MAE com validação TEMPORAL    :", round(mae_cv(cfg, TimeSeriesSplit(5, test_size=12), n - H), 3))
