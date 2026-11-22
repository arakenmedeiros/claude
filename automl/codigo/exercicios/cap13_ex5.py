# Capítulo 13 — solução de referência (EX 5)
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import train_test_split

def ece(p, y, faixas=10):
    f = np.minimum((p * faixas).astype(int), faixas - 1)
    return sum((f == k).mean() * abs(p[f == k].mean() - y[f == k].mean()) for k in range(faixas) if (f == k).any())

X, y = load_breast_cancer(return_X_y=True)
Xa, Xte, ya, yte = train_test_split(X, y, test_size=0.4, stratify=y, random_state=0)
Xtr, Xcal, ytr, ycal = train_test_split(Xa, ya, test_size=0.5, stratify=ya, random_state=0)
rf = RandomForestClassifier(200, random_state=0, n_jobs=-1).fit(Xtr, ytr)
cal = CalibratedClassifierCV(FrozenEstimator(rf), method="sigmoid").fit(Xcal, ycal)
for nome, m in (("floresta bruta", rf), ("floresta + sigmoide", cal)):
    p = m.predict_proba(Xte)[:, 1]
    print(f"{nome:<20s} Brier = {brier_score_loss(yte, p):.4f} | perda log = {log_loss(yte, p):.4f} | ECE = {ece(p, yte):.4f}")

pc, pt = rf.predict_proba(Xcal), rf.predict_proba(Xte); alfa = 0.10
esc = 1 - pc[np.arange(len(ycal)), ycal]; n = len(esc)
q = np.quantile(esc, min(1.0, np.ceil((n + 1) * (1 - alfa)) / n), method="higher")
conj = pt >= 1 - q; tam = conj.sum(1)
print(f"Cobertura no teste = {conj[np.arange(len(yte)), yte].mean():.3f} (alvo {1 - alfa:.2f}) | tamanho médio = {tam.mean():.2f} | abstenção = {(tam == 2).mean():.1%} | conjuntos vazios = {(tam == 0).mean():.1%}")
