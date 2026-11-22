# Capítulo 12 — AutoML para séries temporais, texto e imagens
# Bloco 5. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain

X, Y = make_multilabel_classification(4000, 30, n_classes=6, n_labels=2, length=60, allow_unlabeled=False, random_state=0)
Xa, Xte, Ya, Yte = train_test_split(X, Y, test_size=0.3, random_state=0)
Xtr, Xva, Ytr, Yva = train_test_split(Xa, Ya, test_size=0.3, random_state=1)
print("Frequência de cada rótulo:", Y.mean(0).round(2), "| rótulos por exemplo: %.2f" % Y.sum(1).mean())

def relatorio(nome, P):
    print(f"{nome:<34s} {hamming_loss(Yte, P):9.3f} {f1_score(Yte, P, average='micro'):8.3f} {f1_score(Yte, P, average='macro'):8.3f} {accuracy_score(Yte, P):7.3f}")

print(f"\n{'estratégia (avaliada no teste)':<34s} {'Hamming':>9s} {'F1 micro':>8s} {'F1 macro':>8s} {'exato':>7s}")
fr = Ytr.mean(0); mais_comuns = np.argsort(-fr)[:2]; P0 = np.zeros_like(Yte); P0[:, mais_comuns] = 1
relatorio("linha de base: 2 rótulos mais comuns", P0)
br = OneVsRestClassifier(LogisticRegression(max_iter=2000)).fit(Xtr, Ytr); Pb = br.predict_proba(Xte)
relatorio("relevância binária (limiar 0,5)", (Pb >= 0.5).astype(int))
cadeia = ClassifierChain(LogisticRegression(max_iter=2000), order="random", random_state=0).fit(Xtr, Ytr)
relatorio("cadeia de classificadores", (cadeia.predict_proba(Xte) >= 0.5).astype(int))
Pv = br.predict_proba(Xva); cand = np.linspace(0.05, 0.9, 86)             # um limiar por rótulo, escolhido NA VALIDAÇÃO
t = np.array([cand[int(np.argmax([f1_score(Yva[:, j], (Pv[:, j] >= c).astype(int)) for c in cand]))] for j in range(Y.shape[1])])
relatorio("relevância binária + limiar/rótulo", (Pb >= t).astype(int))
print("Limiares escolhidos por rótulo:", t.round(2))
