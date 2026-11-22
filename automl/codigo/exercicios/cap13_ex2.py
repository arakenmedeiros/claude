# Capítulo 13 — solução de referência (EX 2)
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def gera(n=6000, vies=0.6, p_min=0.3, seed=0):
    r = np.random.default_rng(seed)
    g = (r.random(n) < p_min).astype(int); z = r.normal(0, 1, n)
    renda = np.exp(0.5 * z + 0.35 * r.normal(size=n) - 0.25 * g) * 3.0
    emprego = np.clip(5 + 2 * z + r.normal(0, 2, n), 0, None)
    divida = np.clip(0.4 - 0.1 * z + r.normal(0, 0.1, n), 0, 1)
    cep = 0.8 * g + 0.6 * r.normal(size=n); score = z + r.normal(0, 1, n)
    y_real = (z > 0).astype(int); y_hist = ((z - vies * g + r.normal(0, 0.5, n)) > 0).astype(int)
    return np.c_[renda, emprego, divida, cep, score], g, y_hist, y_real

X, g, yh, yr = gera()
Xtr, Xte, gtr, gte, ytr, yte, yrtr, yrte = train_test_split(X, g, yh, yr, test_size=0.3, random_state=0, stratify=yh)
m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr); p = m.predict(Xte)
for k in (0, 1):
    print(f"grupo {k}: seleção = {p[gte == k].mean():.3f} | TPR real = {p[(gte == k) & (yrte == 1)].mean():.3f} | n = {(gte == k).sum()}")
print("Acurácia contra o histórico: %.3f | contra a verdade: %.3f" % ((p == yte).mean(), (p == yrte).mean()))
