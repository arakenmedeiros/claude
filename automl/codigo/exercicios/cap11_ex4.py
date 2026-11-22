# Capítulo 11 — solução de referência (EX 4)
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
r = np.random.default_rng(1)
arqs = [(tuple(int(r.choice([8, 16, 32, 64, 128])) for _ in range(int(r.integers(1, 4)))), str(r.choice(["relu", "tanh"]))) for _ in range(30)]
def nota(a, it): return MLPClassifier(a[0], activation=a[1], max_iter=it, random_state=0).fit(Xa, ya).score(Xv, yv)
N = {it: np.array([nota(a, it) for a in arqs]) for it in (5, 15, 60)}
print("Spearman  5 x 60 iterações: %.2f" % spearmanr(N[5], N[60])[0])
print("Spearman 15 x 60 iterações: %.2f" % spearmanr(N[15], N[60])[0])
mantidas = np.argsort(-N[5])[:10]
escolha = mantidas[np.argmax(N[60][mantidas])]
print("Melhor das 30 (60 iterações): índice", int(np.argmax(N[60])), "nota %.3f" % N[60].max())
print("Escolha do halving          : índice", int(escolha), "nota %.3f" % N[60][escolha])
print("Custo relativo: %.0f%%" % (100 * (30 * 5 + 10 * 60) / (30 * 60)))
