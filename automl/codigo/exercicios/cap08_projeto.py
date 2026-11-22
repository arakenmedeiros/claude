# Capítulo 8 — solução de referência (PROJ)
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.metrics import r2_score, mean_squared_error

X, y = load_diabetes(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
poly = lambda: PolynomialFeatures(2, include_bias=False)

pipelines = {
    "A: Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "B: quadráticos + Ridge": make_pipeline(poly(), StandardScaler(), Ridge(alpha=10.0)),
    "C: quadráticos + SelectKBest(15) + Ridge": make_pipeline(poly(), StandardScaler(), SelectKBest(f_regression, k=15), Ridge(alpha=1.0)),
    "D: quadráticos + Lasso": make_pipeline(poly(), StandardScaler(), Lasso(alpha=0.5, max_iter=20000)),
    "E: gradient boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=2, random_state=0),
}
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=0)
res = {}
for nome, p in pipelines.items():
    r = cross_val_score(p, Xtr, ytr, cv=cv, scoring="r2")
    res[nome] = r.mean()
    print(f"{nome:42s} R² = {r.mean():.3f} (±{r.std():.3f})")

melhor = max(res, key=res.get)
print("\nMelhor na validação:", melhor)
for nome in dict.fromkeys([melhor, "A: Ridge"]):
    p = pipelines[nome].fit(Xtr, ytr).predict(Xte)
    print(f"Teste final | {nome:42s} R² = {r2_score(yte, p):.3f} | RMSE = {np.sqrt(mean_squared_error(yte, p)):.1f}")
