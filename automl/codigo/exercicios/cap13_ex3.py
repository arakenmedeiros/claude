# Capítulo 13 — solução de referência (EX 3)
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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

NOMES = ["renda", "tempo_emprego", "razao_divida", "cep_score", "score_historico"]
X, g, yh, yr = gera()
Xtr, Xte, gtr, gte = train_test_split(X, g, test_size=0.3, random_state=0, stratify=g)
def auc(cols):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr[:, cols], gtr)
    return roc_auc_score(gte, m.predict_proba(Xte[:, cols])[:, 1])
print("AUC para prever o grupo com todos os atributos: %.3f" % auc(list(range(5))))
for i, nome in enumerate(NOMES): print(f"  sem {nome:<16s} AUC = {auc([j for j in range(5) if j != i]):.3f}")
