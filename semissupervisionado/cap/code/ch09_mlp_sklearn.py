import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

X, y = load_digits(return_X_y=True); X = X / 16.0
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
print("Digits, 50 rótulos, 10 sorteios. MLP de 1 camada escondida (64 neurônios)")
print("modelo                          | acerto no treino | acerto no teste")
def sorteio(s):
    r = np.random.RandomState(s); return np.concatenate([r.choice(np.where(y_tr == d)[0], 5, replace=False) for d in range(10)])
res = {}
for nome, fab in [("regressão logística", lambda: LogisticRegression(max_iter=3000)),
                  ("MLP, alpha = 0,0001 (quase sem L2)", lambda: MLPClassifier((64,), alpha=1e-4, max_iter=2000, random_state=0)),
                  ("MLP, alpha = 0,1", lambda: MLPClassifier((64,), alpha=0.1, max_iter=2000, random_state=0)),
                  ("MLP, alpha = 1", lambda: MLPClassifier((64,), alpha=1.0, max_iter=2000, random_state=0))]:
    tr, te = [], []
    for s in range(10):
        idx = sorteio(s); m = fab().fit(X_tr[idx], y_tr[idx]); tr.append(m.score(X_tr[idx], y_tr[idx])); te.append(m.score(X_te, y_te))
    print(f"{nome:31s} | {100 * np.mean(tr):15.1f}% | {100 * np.mean(te):13.1f}%")
