# Capítulo 12 — solução de referência (PROJ)
import warnings; warnings.filterwarnings("ignore")
import itertools
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit

y = sm.datasets.sunspots.load_pandas().data["SUNACTIVITY"].values.astype(float)
n, H = len(y), 40
MOD = {"ridge": lambda: Ridge(1.0), "floresta": lambda: RandomForestRegressor(100, random_state=0, n_jobs=-1),
       "hgb": lambda: HistGradientBoostingRegressor(max_iter=100, random_state=0)}

def monta(y, L, dif):
    X, a = [], []
    for t in range(L, len(y)):
        b = y[t - 1] if dif else 0.0
        X.append(y[t - L:t][::-1] - b); a.append(y[t] - b)
    return np.array(X), np.array(a)

def mae_cv(cfg):
    mod, L, dif = cfg; X, a = monta(y[:n - H], L, dif)
    return np.mean([np.abs(MOD[mod]().fit(X[tr], a[tr]).predict(X[va]) - a[va]).mean() for tr, va in TimeSeriesSplit(4, test_size=20).split(X)])

def walk(cfg):
    mod, L, dif = cfg; p = []
    for t in range(n - H, n):
        X, a = monta(y[:t], L, dif); m = MOD[mod]().fit(X, a); b = y[t - 1] if dif else 0.0
        p.append(m.predict((y[t - L:t][::-1] - b)[None])[0] + b)
    return np.array(p)

configs = list(itertools.product(MOD, (4, 11, 22), (False, True)))
rank = sorted((mae_cv(c), c) for c in configs)
print("Validação temporal (MAE): 4 melhores e a pior")
for e, c in rank[:4] + rank[-1:]: print(f"  {e:6.2f}  {c[0]:<9s} lags={c[1]:>2d} dif={'sim' if c[2] else 'não'}")
real = y[n - H:]
esc = rank[0][1]; pior = rank[-1][1]
print("\nTESTE (últimos 40 anos, 1 ano à frente), MAE:")
print(f"  mini-AutoML {esc[0]}, lags={esc[1]}, dif={'sim' if esc[2] else 'não'} : {np.abs(walk(esc) - real).mean():.2f}")
print(f"  pior configuração {pior[0]}, lags={pior[1]}, dif={'sim' if pior[2] else 'não'}: {np.abs(walk(pior) - real).mean():.2f}")
print(f"  ingênua                                  : {np.abs(y[n - H - 1:n - 1] - real).mean():.2f}")
print(f"  média dos últimos 11 anos                : {np.abs(np.array([y[t - 11:t].mean() for t in range(n - H, n)]) - real).mean():.2f}")
