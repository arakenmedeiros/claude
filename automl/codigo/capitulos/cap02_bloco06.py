# Capítulo 2 — Aprendizado de máquina em um capítulo
# Bloco 6. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X, y = load_diabetes(return_X_y=True)   # progressão de diabetes em 442 pacientes
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)

for nome, m in [("Sempre a média", DummyRegressor()), ("Regressão Ridge", Ridge(alpha=1.0))]:
    p = m.fit(Xtr, ytr).predict(Xte)
    rmse = np.sqrt(mean_squared_error(yte, p))
    print(f"{nome:16s} MAE={mean_absolute_error(yte, p):6.1f}  RMSE={rmse:6.1f}  R²={r2_score(yte, p):6.3f}")
