# Capítulo 12 — solução de referência (EX 1)
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, TimeSeriesSplit

y = sm.datasets.co2.load_pandas().data["co2"].ffill().resample("MS").mean().values        # preenchimento causal (sem olhar o futuro)
y = y[:-36]
X, a = [], []
for t in range(12, len(y)):
    X.append(np.r_[y[t - 12:t][::-1] - y[t - 12], np.arange(12) == t % 12]); a.append(y[t] - y[t - 12])
X, a = np.array(X), np.array(a)
for nome, cv in (("aleatória", KFold(5, shuffle=True, random_state=0)), ("temporal ", TimeSeriesSplit(5, test_size=12))):
    e = [np.abs(Ridge(1.0).fit(X[tr], a[tr]).predict(X[va]) - a[va]).mean() for tr, va in cv.split(X)]
    print(f"Validação {nome}: MAE = {np.mean(e):.3f}")
