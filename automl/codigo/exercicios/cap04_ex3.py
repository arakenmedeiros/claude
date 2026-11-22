# Capítulo 4 — solução de referência (EX 3)
import numpy as np
from scipy.stats import uniform, loguniform
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

X, y = load_wine(return_X_y=True)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))

def buscar(dist, semente):
    r = RandomizedSearchCV(pipe, {"logisticregression__C": dist}, n_iter=10, cv=cv, random_state=semente).fit(X, y)
    return r.best_score_

for nome, dist in [("uniforme", uniform(0.001, 1000)), ("logarítmico", loguniform(1e-3, 1e3))]:
    notas = [buscar(dist, s) for s in range(10)]
    print(f"{nome:12s} média = {np.mean(notas):.4f} | pior = {np.min(notas):.4f}")
