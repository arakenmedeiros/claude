# Capítulo 12 — AutoML para séries temporais, texto e imagens
# Bloco 2. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
import warnings; warnings.filterwarnings("ignore")
import itertools, json
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
import os; os.makedirs("resultados", exist_ok=True)

bruta = sm.datasets.co2.load_pandas().data["co2"]                     # série semanal, com semanas ausentes
serie = bruta.ffill().resample("MS").mean()                         # preenchimento CAUSAL: usa só o último valor já observado
y = serie.values; n = len(y); H = 36

def monta(y, L, dif):
    X, alvo, base = [], [], []
    for t in range(max(L, 12), len(y)):
        b = y[t - 12] if dif else 0.0
        X.append(np.r_[y[t - L:t][::-1] - b, (np.arange(12) == t % 12)]); alvo.append(y[t] - b); base.append(b)
    return np.array(X), np.array(alvo), np.array(base)

MODELOS = {"ridge": lambda: Ridge(1.0), "hgb": lambda: HistGradientBoostingRegressor(max_iter=100, random_state=0)}

def mae_temporal(cfg):
    mod, L, dif = cfg; X, a, b = monta(y[:n - H], L, dif); e = []
    for tr, va in TimeSeriesSplit(4, test_size=12).split(X):
        e.append(np.abs(MODELOS[mod]().fit(X[tr], a[tr]).predict(X[va]) - a[va]).mean())
    return np.mean(e)

def walk_forward(cfg):
    mod, L, dif = cfg; prev = []
    for t in range(n - H, n):
        X, a, _ = monta(y[:t], L, dif); m = MODELOS[mod]().fit(X, a)
        b = y[t - 12] if dif else 0.0
        x = np.r_[y[t - L:t][::-1] - b, (np.arange(12) == t % 12)]
        prev.append(m.predict(x[None])[0] + b)
    return np.array(prev)

configs = list(itertools.product(MODELOS, (6, 12, 24), (False, True)))
rank = sorted((mae_temporal(c), c) for c in configs)
print("Validação temporal (MAE), melhores e piores configurações:")
for e, c in rank[:3] + rank[-3:]: print(f"  {e:6.3f}  modelo={c[0]:<5s} lags={c[1]:>2d} diferenciação={'sim' if c[2] else 'não'}")

escolhida = rank[0][1]; real = y[n - H:]
prev = walk_forward(escolhida)
ing = y[n - H - 1:n - 1]; saz = y[n - H - 12:n - 12]
print("\nTESTE (últimos 36 meses, 1 mês à frente), MAE:")
print(f"  mini-AutoML ({escolhida[0]}, lags={escolhida[1]}, dif={'sim' if escolhida[2] else 'não'}) : {np.abs(prev - real).mean():.3f}")
print(f"  ingênua (valor do mês anterior)              : {np.abs(ing - real).mean():.3f}")
print(f"  sazonal ingênua (mesmo mês do ano anterior)  : {np.abs(saz - real).mean():.3f}")
pior = rank[-1][1]; prev_p = walk_forward(pior)
print(f"  pior configuração da busca ({pior[0]}, lags={pior[1]}, dif={'sim' if pior[2] else 'não'}) : {np.abs(prev_p - real).mean():.3f}")
json.dump({"hist": y[n - H - 36:n - H].tolist(), "real": real.tolist(), "auto": prev.tolist(), "ing": ing.tolist(), "pior": prev_p.tolist()},
          open("resultados/f12_ts.json", "w"))
