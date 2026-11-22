# Capítulo 14 — solução de referência (EX 2)
import time
import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier

X, y = load_digits(return_X_y=True); lote = np.vstack([X] * 3)[:5000]
print(f"{'árvores':>8s} {'1 previsão (ms)':>16s} {'lote 5 mil (s)':>15s}")
for k in (10, 100, 500):
    m = RandomForestClassifier(k, random_state=0, n_jobs=1).fit(X, y); m.predict(X[:1])
    t = time.perf_counter()
    for _ in range(50): m.predict(X[:1])
    ms = (time.perf_counter() - t) / 50 * 1000
    t = time.perf_counter(); m.predict(lote); print(f"{k:>8d} {ms:>16.2f} {time.perf_counter() - t:>15.3f}")
