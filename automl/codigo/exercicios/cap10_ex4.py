# Capítulo 10 — solução de referência (EX 4)
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from scipy.stats import rankdata, friedmanchisquare
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits, make_classification
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

bases = {"Iris": load_iris(return_X_y=True), "Wine": load_wine(return_X_y=True),
         "Cancer": load_breast_cancer(return_X_y=True), "Digits": load_digits(return_X_y=True),
         "Ruidosa": make_classification(1500, 30, n_informative=8, n_redundant=4, flip_y=0.10, random_state=0)}
modelos = {"Reg. logística": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
           "Floresta": RandomForestClassifier(100, random_state=0, n_jobs=-1),
           "k-vizinhos": make_pipeline(StandardScaler(), KNeighborsClassifier(5))}
cv = StratifiedKFold(5, shuffle=True, random_state=0)
M = np.array([[cross_val_score(m, X, y, cv=cv).mean() for m in modelos.values()] for X, y in bases.values()])
for b, lin in zip(bases, M): print(f"{b:<8s}", "  ".join(f"{v:.3f}" for v in lin))
P = np.array([rankdata(-l) for l in M])
print("Postos médios:", dict(zip(modelos, P.mean(0).round(2))))
est, p = friedmanchisquare(*M.T)
print(f"Friedman: estatística = {est:.2f}, valor-p = {p:.3f}")
