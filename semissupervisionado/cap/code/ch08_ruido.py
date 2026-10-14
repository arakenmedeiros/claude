import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelSpreading, LabelPropagation

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
print("100 rótulos, com uma fração errada; 10 sorteios; acurácia no teste")
print("rótulos errados | Label Propagation | Label Spreading (alpha 0,2) | Label Spreading (alpha 0,8)")
for ruido in [0.0, 0.1, 0.2, 0.3]:
    acc = {"lp": [], "ls2": [], "ls8": []}
    for s in range(10):
        r = np.random.RandomState(s); idx = np.concatenate([r.choice(np.where(y_tr == d)[0], 10, replace=False) for d in range(10)])
        ruim = y_tr[idx].copy(); t = r.rand(len(idx)) < ruido; ruim[t] = (ruim[t] + r.randint(1, 10, t.sum())) % 10
        yp = np.full(len(y_tr), -1); yp[idx] = ruim
        acc["lp"].append(LabelPropagation(kernel="knn", n_neighbors=7, max_iter=1000).fit(X_tr, yp).score(X_te, y_te))
        acc["ls2"].append(LabelSpreading(kernel="knn", n_neighbors=7, alpha=0.2, max_iter=200).fit(X_tr, yp).score(X_te, y_te))
        acc["ls8"].append(LabelSpreading(kernel="knn", n_neighbors=7, alpha=0.8, max_iter=200).fit(X_tr, yp).score(X_te, y_te))
    print(f"{100 * ruido:14.0f}% | {100 * np.mean(acc['lp']):16.1f}% | {100 * np.mean(acc['ls2']):26.1f}% | {100 * np.mean(acc['ls8']):.1f}%")
