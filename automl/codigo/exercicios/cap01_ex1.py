# Capítulo 1 — solução de referência (EX 1)
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

X, y = load_breast_cancer(return_X_y=True)
bobo = DummyClassifier(strategy="most_frequent")
logis = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))

for nome, m in [("Linha de base (bobo)", bobo), ("Regressão logística", logis)]:
    notas = cross_val_score(m, X, y, cv=5)
    print(f"{nome:22s} acurácia = {notas.mean():.3f}")
print("Proporção da classe mais comum:", round(y.mean(), 3))
