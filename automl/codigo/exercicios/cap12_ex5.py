# Capítulo 12 — solução de referência (EX 5)
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

X, Y = make_multilabel_classification(3000, 25, n_classes=5, n_labels=2, length=50, allow_unlabeled=False, random_state=2)
Xa, Xte, Ya, Yte = train_test_split(X, Y, test_size=0.3, random_state=0)
Xtr, Xva, Ytr, Yva = train_test_split(Xa, Ya, test_size=0.3, random_state=1)
br = OneVsRestClassifier(LogisticRegression(max_iter=2000)).fit(Xtr, Ytr); Pv, Pt = br.predict_proba(Xva), br.predict_proba(Xte)
cand = np.linspace(0.05, 0.9, 86)
t = np.array([cand[int(np.argmax([f1_score(Yva[:, j], (Pv[:, j] >= c).astype(int), zero_division=0) for c in cand]))] for j in range(Y.shape[1])])
for nome, P in (("limiar 0,5", (Pt >= 0.5).astype(int)), ("limiar por rótulo", (Pt >= t).astype(int))):
    print(f"{nome:<18s} Hamming = {hamming_loss(Yte, P):.3f} | F1 micro = {f1_score(Yte, P, average='micro'):.3f} | F1 macro = {f1_score(Yte, P, average='macro'):.3f} | exato = {accuracy_score(Yte, P):.3f}")
print("Limiares por rótulo:", t.round(2))
