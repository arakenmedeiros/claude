# Capítulo 11 — solução de referência (EX 2)
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from scipy.stats import spearmanr
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

X, y = load_digits(return_X_y=True)
Xa, Xv, ya, yv = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
sc = StandardScaler().fit(Xa); Xa, Xv = sc.transform(Xa), sc.transform(Xv)
r = np.random.default_rng(0)
res = []
for _ in range(40):
    L = int(r.integers(1, 4)); h = tuple(int(r.choice([8, 16, 32, 64, 128])) for _ in range(L)); ativ = str(r.choice(["relu", "tanh"]))
    d = [64, *h, 10]; params = sum(i * o + o for i, o in zip(d[:-1], d[1:]))
    acc = MLPClassifier(h, activation=ativ, max_iter=40, random_state=0).fit(Xa, ya).score(Xv, yv)
    res.append((acc, params, h, ativ))
rho, _ = spearmanr([p for _, p, _, _ in res], [a for a, _, _, _ in res])
print("Spearman (parâmetros x acurácia): %.2f" % rho)
for acc, params, h, ativ in sorted(res, reverse=True)[:3]: print(f"  {acc:.3f}  {params:>6d} parâmetros  {h} {ativ}")
print("Menor rede entre as 10 melhores:", min(sorted(res, reverse=True)[:10], key=lambda t: t[1])[1], "parâmetros")
