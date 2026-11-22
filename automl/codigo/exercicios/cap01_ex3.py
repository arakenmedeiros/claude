# Capítulo 1 — solução de referência (EX 3)
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

X, y = load_wine(return_X_y=True)

def melhor_da_busca(semente, n=30):
    rng = np.random.default_rng(semente)
    melhor = 0.0
    for _ in range(n):
        alg = rng.choice(["knn", "logistica", "floresta", "svm"])
        if alg == "knn":
            m = make_pipeline(StandardScaler(), KNeighborsClassifier(int(rng.integers(1, 30))))
        elif alg == "logistica":
            m = make_pipeline(StandardScaler(), LogisticRegression(C=10 ** rng.uniform(-3, 3), max_iter=3000))
        elif alg == "floresta":
            m = RandomForestClassifier(int(rng.integers(10, 150)), max_depth=int(rng.integers(2, 15)), random_state=0)
        else:
            m = make_pipeline(StandardScaler(), SVC(C=10 ** rng.uniform(-2, 3), gamma=10 ** rng.uniform(-4, 0)))
        melhor = max(melhor, cross_val_score(m, X, y, cv=5).mean())
    return float(melhor)

notas = [melhor_da_busca(s) for s in range(5)]
print("Melhores notas por semente:", [round(n, 3) for n in notas])
print(f"Média = {np.mean(notas):.3f}  Desvio padrão = {np.std(notas):.3f}")
