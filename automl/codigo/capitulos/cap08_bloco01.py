# Capítulo 8 — Engenharia e seleção automática de atributos
# Bloco 1. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
import numpy as np
import warnings; warnings.filterwarnings("ignore")
from sklearn.datasets import make_circles
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score

X, y = make_circles(400, noise=0.08, factor=0.45, random_state=0)
original = make_pipeline(LogisticRegression())
com_quadrados = make_pipeline(PolynomialFeatures(2, include_bias=False), LogisticRegression(max_iter=2000))

print("Atributos originais        : acurácia =", round(cross_val_score(original, X, y, cv=5).mean(), 3))
print("Com termos quadráticos     : acurácia =", round(cross_val_score(com_quadrados, X, y, cv=5).mean(), 3))
print("Atributos criados por PolynomialFeatures(2):", PolynomialFeatures(2, include_bias=False).fit(X).get_feature_names_out(["x1", "x2"]))
