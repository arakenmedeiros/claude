# Capítulo 13 — Explicabilidade, ética e limitações do AutoML
# Bloco 1. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
cv = StratifiedKFold(10, shuffle=True, random_state=0)
lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000))
hgb = HistGradientBoostingClassifier(random_state=0)
for nome, m in (("Regressão logística (interpretável)", lr), ("Boosting (caixa-preta)", hgb)):
    s = cross_val_score(m, X, y, cv=cv)
    print(f"{nome:<38s} acurácia = {s.mean():.3f} (desvio {s.std():.3f})")
