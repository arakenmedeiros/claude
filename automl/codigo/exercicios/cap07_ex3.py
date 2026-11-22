# Capítulo 7 — solução de referência (EX 3)
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold

X, y = load_breast_cancer(return_X_y=True)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
pontos = []
for C in np.logspace(-3, 2, 15):
    modelo = make_pipeline(StandardScaler(), LogisticRegression(penalty="l1", solver="liblinear", C=C))
    acc = cross_val_score(modelo, X, y, cv=cv).mean()
    nz = int((modelo.fit(X, y)[-1].coef_ != 0).sum())
    pontos.append((nz, acc, C))

domina = lambda a, b: a[0] <= b[0] and a[1] >= b[1] and (a[0] < b[0] or a[1] > b[1])
fronteira = sorted({(nz, round(acc, 4), round(C, 4)) for nz, acc, C in pontos if not any(domina(q, (nz, acc)) for q in pontos)})
print("coeficientes não nulos | acurácia | C")
for nz, acc, C in fronteira:
    print(f"{nz:22d} | {acc:8.3f} | {C}")
print("Acurácia máxima:", round(max(p[1] for p in pontos), 3))
