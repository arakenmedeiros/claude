# Capítulo 13 — solução de referência (EX 1)
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

d = load_breast_cancer(); Xtr, Xte, ytr, yte = train_test_split(d.data, d.target, test_size=0.3, stratify=d.target, random_state=0)
modelos = {"Reg. logística": make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000)), "Boosting": HistGradientBoostingClassifier(random_state=0)}
for nome, m in modelos.items():
    m.fit(Xtr, ytr); pi = permutation_importance(m, Xte, yte, n_repeats=10, random_state=0)
    top = np.argsort(-pi.importances_mean)[:3]
    print(f"{nome:<15s} acurácia = {m.score(Xte, yte):.3f} | top 3:", ", ".join(f"{d.feature_names[i]} ({pi.importances_mean[i]:.3f})" for i in top))
