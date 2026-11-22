import numpy as np
from sklearn.datasets import load_diabetes, make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import kneighbors_graph
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def laprls(Xl, yl, Xu, gamma, gA, gI, k=10):
    """Regressão por mínimos quadrados regularizada pelo Laplaciano (Belkin; Niyogi; Sindhwani, 2006). gI = 0 equivale à regressão ridge com núcleo."""
    A = np.vstack([Xl, Xu]); l, n = len(Xl), len(A); K = rbf_kernel(A, A, gamma=gamma)
    W = kneighbors_graph(A, k, mode="connectivity").toarray(); W = np.maximum(W, W.T); L = np.diag(W.sum(1)) - W
    J = np.zeros((n, n)); J[:l, :l] = np.eye(l); Y = np.r_[yl - yl.mean(), np.zeros(n - l)]
    alfa = np.linalg.solve(J @ K + gA * l * np.eye(n) + gI * l / n ** 2 * L @ K, Y)
    return lambda B: rbf_kernel(B, A, gamma=gamma) @ alfa + yl.mean()

bases = {"Diabetes (UCI/scikit-learn)": load_diabetes(return_X_y=True), "Friedman nº 1 (simulada, 1.000 exemplos)": make_friedman1(1000, noise=1.0, random_state=0)}
print("Regressão com 20 ou 50 rótulos, 20 sorteios. R² no teste (30%) | gamma do núcleo = 1/(número de atributos); gA = 0,01; variamos gI (peso do Laplaciano), sem escolher o melhor no teste")
print("base                                    | rótulos | Ridge linear   | floresta aleatória | núcleo ridge (gI = 0) | LapRLS gI = 1 | gI = 100 | gI = 1000")
for nome, (X, y) in bases.items():
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0); esc = StandardScaler().fit(X_tr); X_tr, X_te = esc.transform(X_tr), esc.transform(X_te); g = 1.0 / X.shape[1]
    for n in [20, 50]:
        R = {k: [] for k in ["ridge", "rf", "krr", "l1", "l100", "l1000"]}
        for s in range(20):
            r = np.random.RandomState(s); idx = r.choice(len(y_tr), n, replace=False); resto = np.setdiff1d(np.arange(len(y_tr)), idx)
            R["ridge"].append(r2_score(y_te, Ridge(alpha=1.0).fit(X_tr[idx], y_tr[idx]).predict(X_te)))
            R["rf"].append(r2_score(y_te, RandomForestRegressor(100, random_state=0).fit(X_tr[idx], y_tr[idx]).predict(X_te)))
            R["krr"].append(r2_score(y_te, laprls(X_tr[idx], y_tr[idx], X_tr[resto], g, 0.01, 0.0)(X_te)))
            for k, gi in [("l1", 1.0), ("l100", 100.0), ("l1000", 1000.0)]: R[k].append(r2_score(y_te, laprls(X_tr[idx], y_tr[idx], X_tr[resto], g, 0.01, gi)(X_te)))
        print(f"{nome:39s} | {n:7d} | " + " | ".join(f"{np.mean(R[k]):6.3f} ± {np.std(R[k]):.3f}" for k in ["ridge", "rf", "krr", "l1", "l100", "l1000"]))
