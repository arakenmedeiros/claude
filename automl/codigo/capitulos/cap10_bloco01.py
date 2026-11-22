# Capítulo 10 — Ferramentas de AutoML na prática
# Bloco 1. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from flaml import AutoML
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

automl = AutoML()
automl.fit(Xtr, ytr, task="classification", time_budget=10, metric="accuracy",
           eval_method="cv", n_splits=3, seed=0, verbose=0)

print("Modelo escolhido        :", automl.best_estimator)
print("Erro de validação (cv)  : %.4f" % automl.best_loss)
print("Acurácia no TESTE final : %.4f" % accuracy_score(yte, automl.predict(Xte)))
print("Melhorias registradas   :", len(automl.config_history))
