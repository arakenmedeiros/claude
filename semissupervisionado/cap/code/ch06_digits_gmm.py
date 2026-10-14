import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.stats import mode

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
Z_tr = PCA(n_components=15, random_state=0).fit(X_tr)
P_tr, P_te = Z_tr.transform(X_tr), Z_tr.transform(X_te)                # 64 atributos -> 15 componentes principais

res = {"regressão logística (só rotulados)": [], "LDA (só rotulados)": [], "GMM semissupervisionado (cluster-e-rotula)": []}
for s in range(10):
    r = np.random.RandomState(s)
    idx = np.concatenate([r.choice(np.where(y_tr == d)[0], 5, replace=False) for d in range(10)])
    res["regressão logística (só rotulados)"].append(LogisticRegression(max_iter=3000).fit(P_tr[idx], y_tr[idx]).score(P_te, y_te))
    res["LDA (só rotulados)"].append(LinearDiscriminantAnalysis().fit(P_tr[idx], y_tr[idx]).score(P_te, y_te))
    # GMM com 10 componentes, médias iniciais = médias dos rotulados de cada dígito; usa TODOS os dados (com e sem rótulo)
    mu0 = np.array([P_tr[idx][y_tr[idx] == d].mean(axis=0) for d in range(10)])
    gmm = GaussianMixture(10, covariance_type="tied", means_init=mu0, reg_covar=1e-2, max_iter=200, random_state=0).fit(P_tr)
    comp = gmm.predict(P_tr[idx]); mapa = {c: mode(y_tr[idx][comp == c], keepdims=False).mode if (comp == c).any() else c for c in range(10)}   # componente -> dígito, pelo voto dos rotulados
    res["GMM semissupervisionado (cluster-e-rotula)"].append(np.mean(np.array([mapa[c] for c in gmm.predict(P_te)]) == y_te))
print("método                                      | acurácia no teste (10 sorteios, 50 rótulos)")
for k, v in res.items(): print(f"{k:43s} | {100 * np.mean(v):5.1f}% ± {100 * np.std(v):.1f}")
