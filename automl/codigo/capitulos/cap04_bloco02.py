# Capítulo 4 — Busca em grade e aleatória
# Bloco 2. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
import numpy as np
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold

X, y = load_digits(return_X_y=True)
X = X / 16.0                                   # pixels vão de 0 a 16: colocamos entre 0 e 1
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

grade = {"C": [0.1, 1, 10, 100], "gamma": [0.0001, 0.001, 0.01, 0.1]}
busca = GridSearchCV(SVC(), grade, cv=cv).fit(X, y)

print("Combinações testadas:", len(busca.cv_results_["params"]))
print("Treinos realizados  :", len(busca.cv_results_["params"]) * 5, "(16 combinações x 5 partes)")
print("Melhor combinação   :", busca.best_params_)
print("Melhor acurácia     :", round(busca.best_score_, 3))
