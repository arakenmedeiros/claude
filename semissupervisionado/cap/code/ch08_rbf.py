import time
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelSpreading

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
r = np.random.RandomState(0); idx = np.concatenate([r.choice(np.where(y_tr == d)[0], 5, replace=False) for d in range(10)])
yp = np.full(len(y_tr), -1); yp[idx] = y_tr[idx]
print("distância quadrada média entre pontos do Digits: %.0f" % np.mean(((X_tr[:200, None, :] - X_tr[None, :200, :]) ** 2).sum(-1)))
print("kernel | parâmetro | acurácia no teste | tempo (s)")
t = time.time(); a = LabelSpreading(kernel="knn", n_neighbors=7, alpha=0.2).fit(X_tr, yp).score(X_te, y_te)
print(f"knn    | k = 7     | {100 * a:16.1f}% | {time.time() - t:.2f}")
for g in [20, 0.1, 0.02, 0.005, 0.001]:
    t = time.time(); a = LabelSpreading(kernel="rbf", gamma=g, alpha=0.2, max_iter=100).fit(X_tr, yp).score(X_te, y_te)
    print(f"rbf    | gamma = {g:<6} | {100 * a:16.1f}% | {time.time() - t:.2f}")
