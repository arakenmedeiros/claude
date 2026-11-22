# Capítulo 2 — solução de referência (EX 1)
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
tamanhos = [0.1, 0.3, 0.6, 1.0]
modelos = {"Árvore sem limite": DecisionTreeClassifier(random_state=0),
           "Regressão logística": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))}
for nome, m in modelos.items():
    n, tr, va = learning_curve(m, X, y, train_sizes=tamanhos, cv=5, scoring="accuracy")
    print(nome)
    for ni, a, b in zip(n, tr.mean(1), va.mean(1)):
        print(f"  treino com {ni:3d} exemplos: treino={a:.3f}  validação={b:.3f}  distância={a-b:.3f}")
