# Capítulo 2 — Aprendizado de máquina em um capítulo
# Bloco 2. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import validation_curve
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)
profundidades = [1, 2, 3, 5, 8, 12]
treino, valid = validation_curve(DecisionTreeClassifier(random_state=0), X, y,
                                 param_name="max_depth", param_range=profundidades, cv=5)

print("profundidade | acurácia no treino | acurácia na validação")
for p, t, v in zip(profundidades, treino.mean(axis=1), valid.mean(axis=1)):
    print(f"{p:12d} | {t:18.3f} | {v:21.3f}")
