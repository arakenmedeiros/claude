# Capítulo 1 — solução de referência (PROJ)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

def sortear(rng):
    alg = rng.choice(["knn", "logistica", "floresta", "svm"])
    if alg == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(int(rng.integers(1, 30))))
    if alg == "logistica":
        return make_pipeline(StandardScaler(), LogisticRegression(C=10 ** rng.uniform(-3, 3), max_iter=3000))
    if alg == "floresta":
        return RandomForestClassifier(int(rng.integers(10, 150)), max_depth=int(rng.integers(2, 15)), random_state=0)
    return make_pipeline(StandardScaler(), SVC(C=10 ** rng.uniform(-2, 3), gamma=10 ** rng.uniform(-4, 0)))

def buscar(orcamento, semente):
    rng = np.random.default_rng(semente)
    melhor, melhor_m = -1.0, None
    for _ in range(orcamento):
        m = sortear(rng)
        nota = cross_val_score(m, Xtr, ytr, cv=5).mean()
        if nota > melhor:
            melhor, melhor_m = nota, m
    melhor_m.fit(Xtr, ytr)
    return float(melhor), float(melhor_m.score(Xte, yte))

orcamentos = [5, 10, 20, 40, 80]
val, teste = [], []
for o in orcamentos:
    r = [buscar(o, s) for s in range(5)]
    val.append(np.mean([a for a, _ in r])); teste.append(np.mean([b for _, b in r]))
    print(f"orçamento {o:3d}: validação = {val[-1]:.3f} | teste final = {teste[-1]:.3f}")

plt.plot(orcamentos, val, "o-", label="validação cruzada (usada na busca)")
plt.plot(orcamentos, teste, "s-", label="teste final (nunca visto)")
plt.xscale("log"); plt.xlabel("nº de tentativas"); plt.ylabel("acurácia"); plt.legend()
plt.title("Mais busca ajuda? Validação vs. teste"); plt.show()
