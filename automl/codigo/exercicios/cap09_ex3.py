# Capítulo 9 — solução de referência (EX 3)
import itertools
import numpy as np
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=2000, n_features=20, n_informative=8, n_redundant=4, n_classes=3,
                           n_clusters_per_class=3, class_sep=0.9, flip_y=0.08, random_state=1)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

def discordancia(modelos):
    prev = [m.fit(Xtr, ytr).predict(Xte) for m in modelos]
    return np.mean([np.mean(a != b) for a, b in itertools.combinations(prev, 2)])

floresta = lambda s: RandomForestClassifier(100, random_state=s, n_jobs=-1)
tres_florestas = [("rf%d" % s, floresta(s)) for s in (1, 2, 3)]
quatro_tipos = [("svm", SVC(C=10, gamma=0.02, probability=True, random_state=0)), ("knn", KNeighborsClassifier(5, weights="distance")),
                ("lr", LogisticRegression(C=1, max_iter=2000)), ("rf", floresta(0))]

print(f"(a) uma floresta                : teste = {floresta(0).fit(Xtr, ytr).score(Xte, yte):.3f}")
print(f"(b) três florestas (sementes)   : teste = {VotingClassifier(tres_florestas, voting='soft').fit(Xtr, ytr).score(Xte, yte):.3f} | discordância média = {discordancia([m for _, m in tres_florestas]):.3f}")
print(f"(c) quatro tipos de modelos     : teste = {VotingClassifier(quatro_tipos, voting='soft').fit(Xtr, ytr).score(Xte, yte):.3f} | discordância média = {discordancia([m for _, m in quatro_tipos]):.3f}")
