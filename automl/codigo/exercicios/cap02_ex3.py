# Capítulo 2 — solução de referência (EX 3)
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_wine(return_X_y=True)
modelo = make_pipeline(StandardScaler(), KNeighborsClassifier())

notas = []
for semente in range(30):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=semente)
    notas.append(modelo.fit(Xtr, ytr).score(Xte, yte))
print(f"30 divisões: menor={min(notas):.3f}  maior={max(notas):.3f}  média={np.mean(notas):.3f}")

cv = cross_val_score(modelo, X, y, cv=5)
print(f"Validação cruzada (5 partes): média={cv.mean():.3f}  desvio={cv.std():.3f}")
