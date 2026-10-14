import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier

def tri_training(X, y_parcial, fabrica, rodadas=5, semente=0):
    """Tri-training simplificado (Zhou e Li, 2005): três classificadores treinados em amostras bootstrap dos rotulados.
    Um exemplo sem rótulo recebe pseudo-rótulo para o classificador i quando os outros dois concordam."""
    rng = np.random.RandomState(semente)
    com = np.where(y_parcial != -1)[0]; sem = np.where(y_parcial == -1)[0]
    H = []
    for _ in range(3):
        b = rng.choice(com, len(com), replace=True)
        H.append(fabrica().fit(X[b], y_parcial[b]))
    for _ in range(rodadas):
        pred = [h.predict(X[sem]) for h in H]
        novos = []
        for i in range(3):
            j, k = [m for m in range(3) if m != i]
            conc = pred[j] == pred[k]                                   # os outros dois concordam
            Xi = np.vstack([X[com], X[sem][conc]]); yi = np.r_[y_parcial[com], pred[j][conc]]
            novos.append((Xi, yi))
        H = [fabrica().fit(Xi, yi) for Xi, yi in novos]
    return H

def votar(H, X):
    votos = np.array([h.predict(X) for h in H])
    return np.array([np.bincount(v, minlength=10).argmax() for v in votos.T])

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
fab = lambda: LogisticRegression(max_iter=2000)
res = {"só rotulados": [], "self-training": [], "tri-training": []}
for s in range(10):
    r = np.random.RandomState(s)
    idx = np.concatenate([r.choice(np.where(y_tr == d)[0], 5, replace=False) for d in range(10)])
    yp = np.full(len(y_tr), -1); yp[idx] = y_tr[idx]
    res["só rotulados"].append(fab().fit(X_tr[idx], y_tr[idx]).score(X_te, y_te))
    res["self-training"].append(SelfTrainingClassifier(fab(), threshold=0.9).fit(X_tr, yp).score(X_te, y_te))
    res["tri-training"].append(np.mean(votar(tri_training(X_tr, yp, fab, semente=s), X_te) == y_te))
for k, v in res.items(): print(f"{k:14s}: {100 * np.mean(v):5.1f}% ± {100 * np.std(v):.1f}")
