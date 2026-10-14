import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
configs = {
    "limiar 0,9 (padrão do livro)":       dict(criterion="threshold", threshold=0.9, max_iter=10),
    "limiar 0,75 (padrão do sklearn)":    dict(criterion="threshold", threshold=0.75, max_iter=10),
    "k_best = 10 por rodada":             dict(criterion="k_best", k_best=10, max_iter=100),
    "k_best = 50 por rodada":             dict(criterion="k_best", k_best=50, max_iter=100),
    "k_best = 200 por rodada":            dict(criterion="k_best", k_best=200, max_iter=100),
}
print("critério                         | rodadas | exemplos que ganharam rótulo | acurácia no teste")
for nome, kw in configs.items():
    rod, rot, acc = [], [], []
    for s in range(10):
        r = np.random.RandomState(s)
        idx = np.concatenate([r.choice(np.where(y_tr == d)[0], 5, replace=False) for d in range(10)])
        yp = np.full(len(y_tr), -1); yp[idx] = y_tr[idx]
        st = SelfTrainingClassifier(LogisticRegression(max_iter=2000), **kw).fit(X_tr, yp)
        rod.append(st.n_iter_); rot.append((st.labeled_iter_ > 0).sum()); acc.append(st.score(X_te, y_te))
    print(f"{nome:32s} | {np.mean(rod):7.1f} | {np.mean(rot):28.0f} | {100 * np.mean(acc):16.1f}%")
