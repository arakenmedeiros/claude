# Capítulo 8 — solução de referência (EX 2)
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold

X0, y = load_breast_cancer(return_X_y=True)
rng = np.random.default_rng(0)
X = np.hstack([X0, rng.normal(size=(len(y), 200))])                  # 30 reais + 200 de ruído
cv = StratifiedKFold(5, shuffle=True, random_state=0)

print("k    | acurácia")
for k in (2, 5, 10, 20, 30, 50, 100, 230):
    pipe = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=k), LogisticRegression(max_iter=3000))
    print(f"{k:4d} | {cross_val_score(pipe, X, y, cv=cv).mean():.3f}")
sem = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
print("sem seleção (230 atributos):", round(cross_val_score(sem, X, y, cv=cv).mean(), 3))
