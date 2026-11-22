# Capítulo 4 — Busca em grade e aleatória
# Bloco 3. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
# --- blocos anteriores deste capítulo (encadeados) ---
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
# --- fim dos blocos anteriores ---
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV

espaco = {"C": loguniform(1e-2, 1e3), "gamma": loguniform(1e-5, 1)}
aleatoria = RandomizedSearchCV(SVC(), espaco, n_iter=16, cv=cv, random_state=0).fit(X, y)

print("Combinações testadas:", len(aleatoria.cv_results_["params"]), "(mesmo orçamento da grade)")
print("Melhor acurácia     :", round(aleatoria.best_score_, 3))
print("Melhor C e gamma    : C = %.2f, gamma = %.4f" % (aleatoria.best_params_["C"], aleatoria.best_params_["gamma"]))
