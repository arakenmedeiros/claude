# Capítulo 4 — solução de referência (EX 2)
import numpy as np
from scipy.stats import randint, uniform
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

X, y = load_breast_cancer(return_X_y=True)
cv = StratifiedKFold(5, shuffle=True, random_state=0)

grade = {"n_estimators": [20, 60], "max_depth": [3, None], "max_features": ["sqrt", 0.5]}
g = GridSearchCV(RandomForestClassifier(random_state=0), grade, cv=cv).fit(X, y)
print(f"Grade ({len(g.cv_results_['params'])} combinações): melhor = {g.best_score_:.4f}")

espaco = {"n_estimators": randint(10, 80), "max_depth": randint(2, 12), "max_features": uniform(0.05, 0.9)}
notas = [RandomizedSearchCV(RandomForestClassifier(random_state=0), espaco, n_iter=8, cv=cv, random_state=s).fit(X, y).best_score_
         for s in range(4)]
print("Aleatória (8 tentativas) x 4 sementes:", np.round(notas, 4), "| média =", round(np.mean(notas), 4))
