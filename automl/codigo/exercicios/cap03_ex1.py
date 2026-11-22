# Capítulo 3 — solução de referência (EX 1)
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

X, y = load_breast_cancer(return_X_y=True)
rng = np.random.default_rng(0)
X = X.copy(); X[rng.random(X.shape) < 0.15] = np.nan
print("Fração de ausentes:", round(np.isnan(X).mean(), 3))

imputadores = {
    "média": SimpleImputer(strategy="mean"),
    "mediana": SimpleImputer(strategy="median"),
    "k-vizinhos": KNNImputer(n_neighbors=5),
    "mediana + indicador": SimpleImputer(strategy="median", add_indicator=True),
}
cv = StratifiedKFold(5, shuffle=True, random_state=0)
for nome, imp in imputadores.items():
    pipe = make_pipeline(imp, StandardScaler(), LogisticRegression(max_iter=3000))
    n = cross_val_score(pipe, X, y, cv=cv)
    print(f"{nome:20s} acurácia = {n.mean():.3f} (±{n.std():.3f})")
