# Capítulo 2 — solução de referência (EX 5)
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(4000, 15, n_informative=5, weights=[0.96, 0.04], flip_y=0.01, class_sep=0.8, random_state=3)
Xa, Xte, ya, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
Xtr, Xva, ytr, yva = train_test_split(Xa, ya, test_size=0.3, stratify=ya, random_state=1)
modelo = lambda **kw: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, **kw))
m0, mw = modelo().fit(Xtr, ytr), modelo(class_weight="balanced").fit(Xtr, ytr)
cand = np.linspace(0.02, 0.9, 89); pv = m0.predict_proba(Xva)[:, 1]
t = cand[int(np.argmax([f1_score(yva, (pv >= c).astype(int)) for c in cand]))]
for nome, pred in (("(a) limiar 0,5", m0.predict(Xte)), ("(b) pesos balanceados", mw.predict(Xte)), (f"(c) limiar {t:.2f} (validação)", (m0.predict_proba(Xte)[:, 1] >= t).astype(int))):
    print(f"{nome:<26s} precisão = {precision_score(yte, pred, zero_division=0):.3f} | revocação = {recall_score(yte, pred):.3f} | F1 = {f1_score(yte, pred):.3f}")
