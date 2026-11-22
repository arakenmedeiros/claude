# Capítulo 2 — solução de referência (EX 4)
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
rng = np.random.default_rng(0)
y = rng.permutation(y)                                  # destrói a relação entre X e y
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, stratify=y, random_state=0)

melhor, melhor_m = 0.0, None
for _ in range(100):
    if rng.random() < 0.5:
        m = make_pipeline(StandardScaler(), KNeighborsClassifier(int(rng.integers(1, 40))))
    else:
        m = DecisionTreeClassifier(max_depth=int(rng.integers(1, 12)), random_state=0)
    nota = cross_val_score(m, Xtr, ytr, cv=5).mean()
    if nota > melhor:
        melhor, melhor_m = nota, m
melhor_m.fit(Xtr, ytr)
print(f"Melhor nota de validação (100 candidatos): {melhor:.3f}")
print(f"Nota no teste separado                    : {melhor_m.score(Xte, yte):.3f}")
print(f"Proporção da classe majoritária (linha de base): {max(y.mean(), 1-y.mean()):.3f}")
