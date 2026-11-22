# Capítulo 10 — solução de referência (PROJ)
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from flaml import AutoML
from scipy.stats import rankdata, friedmanchisquare
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

bases = {"Iris": load_iris(return_X_y=True), "Wine": load_wine(return_X_y=True),
         "Cancer": load_breast_cancer(return_X_y=True), "Digits": load_digits(return_X_y=True)}
nomes = ["Floresta", "Boosting", "FLAML(20)"]
M = []; escolhas = {}
for b, (X, y) in bases.items():
    acc = {n: [] for n in nomes}; esc = []
    for s in (0, 1, 2):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=s)
        m1 = RandomForestClassifier(200, random_state=s, n_jobs=-1).fit(Xtr, ytr)
        m2 = HistGradientBoostingClassifier(random_state=s).fit(Xtr, ytr)
        a = AutoML(); a.fit(Xtr, ytr, task="classification", time_budget=120, max_iter=20, metric="accuracy",
                            eval_method="cv", n_splits=3, seed=s, verbose=0)
        for n, m in zip(nomes, (m1, m2, a)): acc[n].append((m.predict(Xte) == yte).mean())
        esc.append((a.best_estimator, len(a.config_history)))
    M.append([np.mean(acc[n]) for n in nomes]); escolhas[b] = esc
M = np.array(M)
print(f"{'':<8s}" + "  ".join(f"{n:>10s}" for n in nomes))
for b, lin in zip(bases, M): print(f"{b:<8s}" + "  ".join(f"{v:10.3f}" for v in lin))
P = np.array([rankdata(-l) for l in M])
print("Postos médios:", dict(zip(nomes, P.mean(0).round(2))))
est, p = friedmanchisquare(*M.T)
print(f"Friedman: estatística = {est:.2f}, valor-p = {p:.3f} (N = {len(M)})")
print("Escolhas do FLAML (algoritmo, nº de configurações):", {b: e for b, e in escolhas.items()})
