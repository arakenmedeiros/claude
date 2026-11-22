import time
import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn.semi_supervised import LabelSpreading
from utilidades import carregar_mnist

X, y, _, _ = carregar_mnist(); X = X.flatten(1).numpy(); y = y.numpy()
print("MNIST, PCA de 50 componentes, 100 rótulos sorteados entre os n exemplos. Grafo de k vizinhos (matriz esparsa) x kernel RBF (matriz densa)")
print("     n | tempo do grafo | tempo do Label Spreading (knn) | arestas | memória do grafo esparso | memória de uma matriz densa n x n | tempo do Label Spreading (RBF)")
for n in [2000, 5000, 10000, 20000]:
    Z = PCA(50, random_state=0).fit_transform(X[:n]); yy = y[:n]; r = np.random.RandomState(0); idx = r.choice(n, 100, replace=False); yp = np.full(n, -1); yp[idx] = yy[idx]
    t = time.perf_counter(); G = kneighbors_graph(Z, 10); tg = time.perf_counter() - t
    t = time.perf_counter(); LabelSpreading(kernel="knn", n_neighbors=10, alpha=0.2, max_iter=30).fit(Z, yp); ts = time.perf_counter() - t
    mem = (G.data.nbytes + G.indices.nbytes + G.indptr.nbytes) / 1e6; densa = n * n * 8 / 1e6
    if n <= 5000:
        t = time.perf_counter(); LabelSpreading(kernel="rbf", gamma=0.01, alpha=0.2, max_iter=30).fit(Z, yp); tr = f"{time.perf_counter() - t:.1f} s"
    else: tr = "não executado"
    print(f"{n:6d} | {tg:11.2f} s | {ts:26.2f} s | {G.nnz:7d} | {mem:19.1f} MB | {densa:24.0f} MB | {tr}")
print("\nDiagnóstico de componentes sem nenhum rótulo (n = 10.000, 100 rótulos), conforme o número de vizinhos k")
print("k  | componentes | nós em componentes sem nenhum rotulado")
n = 10000; Z = PCA(50, random_state=0).fit_transform(X[:n]); r = np.random.RandomState(0); idx = r.choice(n, 100, replace=False)
for k in [1, 2, 3, 5, 10]:
    G = kneighbors_graph(Z, k); G = G.maximum(G.T); nc, lab = connected_components(G, directed=False); com_rot = np.unique(lab[idx]); sem = ~np.isin(lab, com_rot)
    print(f"{k:2d} | {nc:11d} | {100 * sem.mean():.1f}% ({sem.sum()} nós)")
