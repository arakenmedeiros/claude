import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph
from sklearn.semi_supervised import LabelSpreading

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
print("k  | componentes | arestas que ligam a mesma classe | acurácia (50 rótulos, 10 sorteios)")
for k in [3, 5, 7, 10, 15, 25, 50]:
    G = kneighbors_graph(X_tr, k, mode="connectivity"); G = G.maximum(G.T)
    n_comp = connected_components(G, directed=False)[0]
    i, j = G.nonzero(); pureza = np.mean(y_tr[i] == y_tr[j])
    acc = []
    for s in range(10):
        r = np.random.RandomState(s); idx = np.concatenate([r.choice(np.where(y_tr == d)[0], 5, replace=False) for d in range(10)])
        yp = np.full(len(y_tr), -1); yp[idx] = y_tr[idx]
        acc.append(LabelSpreading(kernel="knn", n_neighbors=k, alpha=0.2, max_iter=200).fit(X_tr, yp).score(X_te, y_te))
    print(f"{k:2d} | {n_comp:11d} | {100 * pureza:31.1f}% | {100 * np.mean(acc):.1f}%")
