import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelSpreading

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
ks = [5, 7, 10, 15, 25]; alphas = [0.05, 0.2, 0.5, 0.8, 0.95]
print("Digits, 50 rótulos, 10 sorteios. Linhas: n_neighbors; colunas: alpha. Acurácia no teste (%).")
print("  k \\ alpha | " + " | ".join(f"{a:5.2f}" for a in alphas))
for k in ks:
    linha = []
    for a in alphas:
        acc = []
        for s in range(10):
            r = np.random.RandomState(s); idx = np.concatenate([r.choice(np.where(y_tr == d)[0], 5, replace=False) for d in range(10)])
            yp = np.full(len(y_tr), -1); yp[idx] = y_tr[idx]
            acc.append(LabelSpreading(kernel="knn", n_neighbors=k, alpha=a, max_iter=300).fit(X_tr, yp).score(X_te, y_te))
        linha.append(100 * np.mean(acc))
    print(f"  {k:9d} | " + " | ".join(f"{v:5.1f}" for v in linha))
