import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

print("rótulos errados | só rotulados | self-training | ganho (p.p.)")
for ruido in [0.0, 0.1, 0.2, 0.3]:
    a, b = [], []
    for s in range(10):
        r = np.random.RandomState(s)
        idx = np.concatenate([r.choice(np.where(y_tr == d)[0], 10, replace=False) for d in range(10)])   # 100 rótulos
        y_ruim = y_tr[idx].copy()
        trocar = r.rand(len(idx)) < ruido
        y_ruim[trocar] = (y_ruim[trocar] + r.randint(1, 10, trocar.sum())) % 10     # troca por outra classe
        yp = np.full(len(y_tr), -1); yp[idx] = y_ruim
        a.append(LogisticRegression(max_iter=2000).fit(X_tr[idx], y_ruim).score(X_te, y_te))
        b.append(SelfTrainingClassifier(LogisticRegression(max_iter=2000), threshold=0.9).fit(X_tr, yp).score(X_te, y_te))
    print(f"{100 * ruido:14.0f}% | {100 * np.mean(a):11.1f}% | {100 * np.mean(b):12.1f}% | {100 * (np.mean(b) - np.mean(a)):+11.1f}")
