# Capítulo 12 — solução de referência (EX 4)
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from scipy.ndimage import shift, rotate
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, stratify=y, random_state=0)
Xtr, ytr = Xtr[:300], ytr[:300]

def distorce(Xs, r, dmax, rot, ruido):
    out = []
    for x in Xs:
        im = shift(x.reshape(8, 8), r.uniform(-dmax, dmax, 2), order=1, mode="constant")
        if rot > 0: im = rotate(im, r.uniform(-rot, rot), reshape=False, order=1, mode="constant")
        out.append((im + r.normal(0, ruido, im.shape)).ravel())
    return np.array(out)

Xte_d = distorce(Xte, np.random.default_rng(123), 1.0, 10, 0.0)
politicas = {"sem ampliação": (0, 0, 0), "só deslocamento": (1.5, 0, 0), "só rotação": (0, 15, 0), "só ruído": (0, 0, 0.8), "as três": (1.5, 15, 0.8)}
for nome, (d, rot, ru) in politicas.items():
    r = np.random.default_rng(0); Xs, ys = [Xtr], [ytr]
    if max(d, rot, ru) > 0:
        for _ in range(4): Xs.append(distorce(Xtr, r, d, rot, ru)); ys.append(ytr)
    m = SVC(C=10, gamma="scale").fit(np.vstack(Xs), np.concatenate(ys))
    print(f"{nome:<16s} teste limpo = {m.score(Xte, yte):.3f} | teste distorcido = {m.score(Xte_d, yte):.3f}")
