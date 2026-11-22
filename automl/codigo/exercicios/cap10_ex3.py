# Capítulo 10 — solução de referência (EX 3)
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from flaml import AutoML
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

X, y = make_classification(1500, 30, n_informative=8, n_redundant=4, flip_y=0.10, random_state=0)
import time
R = {"Reg. logística": [], "Floresta padrão": [], "FLAML (5 s)": []}; T = {k: [] for k in R}
for s in range(5):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=s)
    for nome in R:
        t0 = time.perf_counter()
        if nome == "Reg. logística": m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr)
        elif nome == "Floresta padrão": m = RandomForestClassifier(200, random_state=s, n_jobs=1).fit(Xtr, ytr)
        else:
            m = AutoML(); m.fit(Xtr, ytr, task="classification", time_budget=5, metric="accuracy", eval_method="cv", n_splits=3, seed=s, n_jobs=1, verbose=0)
        T[nome].append(time.perf_counter() - t0); R[nome].append((m.predict(Xte) == yte).mean())
for nome, v in R.items(): print(f"{nome:<16s} média = {np.mean(v):.3f}  desvio = {np.std(v):.3f} | tempo médio = {np.mean(T[nome]):5.2f} s (1 núcleo)")
