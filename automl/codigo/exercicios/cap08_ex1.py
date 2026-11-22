# Capítulo 8 — solução de referência (EX 1)
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold

rng = np.random.default_rng(0)
x1, x2 = rng.normal(size=1000), rng.normal(size=1000)
y = x1 * x2 + rng.normal(0, 0.3, 1000)

cv = KFold(5, shuffle=True, random_state=0)
a = cross_val_score(LinearRegression(), np.c_[x1, x2], y, cv=cv, scoring="r2").mean()
b = cross_val_score(LinearRegression(), np.c_[x1, x2, x1 * x2], y, cv=cv, scoring="r2").mean()
print(f"(a) só x1 e x2      : R² = {a:.3f}")
print(f"(b) com x1·x2       : R² = {b:.3f}")
