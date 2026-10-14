import os, gzip, pickle, time, urllib.request
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import LabelSpreading

def baixar(url, destino):
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    if not os.path.exists(destino): urllib.request.urlretrieve(url, destino)
    return destino

arq = baixar("https://raw.githubusercontent.com/mnielsen/neural-networks-and-deep-learning/master/data/mnist.pkl.gz", "dados/mnist.pkl.gz")   # espelho do MNIST
(Xa, ya), _, (Xt, yt) = pickle.load(gzip.open(arq), encoding="latin1")
rng = np.random.RandomState(0); sub = rng.choice(len(Xa), 10000, replace=False)
X, y = Xa[sub], ya[sub]                                                  # 10.000 imagens para o treino (a maioria sem rótulo)
print("treino:", X.shape, "| teste:", Xt.shape, "| rótulos usados: 100 (10 por dígito)")

def rodar(Z, Zt, s, k=10, alpha=0.2):
    r = np.random.RandomState(s); idx = np.concatenate([r.choice(np.where(y == d)[0], 10, replace=False) for d in range(10)])
    yp = np.full(len(y), -1); yp[idx] = y[idx]
    ls = LabelSpreading(kernel="knn", n_neighbors=k, alpha=alpha, max_iter=100).fit(Z, yp)
    sem = yp == -1
    return (LogisticRegression(max_iter=3000).fit(Z[idx], y[idx]).score(Zt, yt), KNeighborsClassifier(1).fit(Z[idx], y[idx]).score(Zt, yt),
            np.mean(ls.transduction_[sem] == y[sem]), ls.score(Zt, yt))

print("\ndimensões | regressão logística | 1-vizinho (só rotulados) | LS nos sem rótulo (transdutivo) | LS no teste (indutivo) | tempo (s)")
for nome, dim in [("784 (pixels)", None), ("PCA 50", 50), ("PCA 20", 20), ("PCA 10", 10)]:
    t = time.time()
    if dim: p = PCA(dim, random_state=0).fit(X); Z, Zt = p.transform(X), p.transform(Xt)
    else: Z, Zt = X, Xt
    res = np.array([rodar(Z, Zt, s) for s in range(3)])
    m = 100 * res.mean(axis=0)
    print(f"{nome:12s} | {m[0]:18.1f}% | {m[1]:23.1f}% | {m[2]:30.1f}% | {m[3]:21.1f}% | {(time.time() - t) / 3:.1f}")
