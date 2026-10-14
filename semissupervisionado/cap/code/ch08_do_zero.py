import numpy as np
from scipy.sparse import csr_matrix
from sklearn.datasets import make_moons
from sklearn.neighbors import kneighbors_graph
from sklearn.semi_supervised import LabelSpreading, LabelPropagation

def grafo_knn(X, k=7):
    """Matriz de pesos W: cada ponto liga-se aos seus k vizinhos mais próximos (grafo simétrico, pesos 1)."""
    W = kneighbors_graph(X, k, mode="connectivity", include_self=False).toarray()
    return np.maximum(W, W.T)

def propagacao(W, y_parcial, n_classes, iters=200):
    """Label Propagation: a cada passo, cada ponto recebe a média dos rótulos dos vizinhos; os rotulados voltam ao valor original."""
    P = W / W.sum(axis=1, keepdims=True)                      # matriz de transição (linhas somam 1)
    rot = y_parcial != -1
    Y = np.zeros((len(W), n_classes)); Y[rot, y_parcial[rot]] = 1
    F = Y.copy(); F[~rot] = 1 / n_classes
    historico = []
    for _ in range(iters):
        F = P @ F
        F[rot] = Y[rot]                                       # "clamping": os rótulos conhecidos nunca mudam
        historico.append(F.argmax(axis=1))
    return F, historico

def espalhamento(W, y_parcial, n_classes, alpha=0.8, iters=200):
    """Label Spreading (Zhou et al., 2003): F <- alpha * S F + (1 - alpha) * Y, com S = D^(-1/2) W D^(-1/2). Os rótulos podem mudar."""
    d = W.sum(axis=1); S = W / np.sqrt(np.outer(d, d))
    rot = y_parcial != -1
    Y = np.zeros((len(W), n_classes)); Y[rot, y_parcial[rot]] = 1
    F = Y.copy()
    for _ in range(iters): F = alpha * (S @ F) + (1 - alpha) * Y
    return F

X, y = make_moons(300, noise=0.07, random_state=2)
def mais_proximo(p, c):
    i = np.where(y == c)[0]; return i[np.argmin(((X[i] - p) ** 2).sum(1))]
idx = [mais_proximo((0, 1.0), 0), mais_proximo((-0.8, 0.5), 0), mais_proximo((1, -0.5), 1), mais_proximo((1.8, -0.1), 1)]
yp = np.full(len(y), -1); yp[idx] = y[idx]

W = grafo_knn(X, 7)
F, hist = propagacao(W, yp, 2)
print("Label Propagation do zero: acurácia a cada passo")
for t in [1, 3, 5, 10, 20, 50, 200]:
    print(f"  passo {t:3d}: {100 * np.mean(hist[t - 1] == y):5.1f}%")
Fs = espalhamento(W, yp, 2)
print(f"Label Spreading do zero (alpha = 0,8): {100 * np.mean(Fs.argmax(axis=1) == y):.1f}%")

lp = LabelPropagation(kernel="knn", n_neighbors=7, max_iter=2000).fit(X, yp)
ls = LabelSpreading(kernel="knn", n_neighbors=7, alpha=0.8, max_iter=2000).fit(X, yp)
print(f"sklearn LabelPropagation              : {100 * np.mean(lp.transduction_ == y):.1f}%")
print(f"sklearn LabelSpreading (alpha = 0,8)  : {100 * np.mean(ls.transduction_ == y):.1f}%")
print("concordância dos rótulos, minha versão x sklearn:  propagation %.1f%% | spreading %.1f%%" % (100 * np.mean(F.argmax(1) == lp.transduction_), 100 * np.mean(Fs.argmax(1) == ls.transduction_)))
