# Capítulo 14 — MLOps, implantação e projeto integrador
# Bloco 2. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
import warnings; warnings.filterwarnings("ignore")
import time
import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_digits(return_X_y=True)
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
rf = RandomForestClassifier(300, random_state=0, n_jobs=1)
hgb = HistGradientBoostingClassifier(random_state=0)
vot = VotingClassifier([("lr", lr), ("rf", RandomForestClassifier(300, random_state=0, n_jobs=1)), ("svm", make_pipeline(StandardScaler(), SVC(probability=True, random_state=0)))], voting="soft")
modelos = {"Regressão logística": lr, "Floresta (300 árvores)": rf, "Boosting": hgb, "Ensemble de 3 modelos": vot}
lote = np.vstack([X] * 6)[:10000]
print(f"{'modelo':<26s} {'1 previsão (ms)':>16s} {'lote de 10 mil (s)':>20s}")
for nome, m in modelos.items():
    m.fit(X, y); um = X[:1]; m.predict(um)
    t = time.perf_counter()
    for _ in range(50): m.predict(um)
    um_ms = (time.perf_counter() - t) / 50 * 1000
    t = time.perf_counter(); m.predict(lote); tl = time.perf_counter() - t
    print(f"{nome:<26s} {um_ms:>16.2f} {tl:>20.2f}")
