# Capítulo 3 — solução de referência (EX 2)
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

rng = np.random.default_rng(0)
cidades = np.array(list("ABCDE"))
taxa = {"A": 0.10, "B": 0.80, "C": 0.20, "D": 0.90, "E": 0.15}
X = rng.choice(cidades, size=(2000, 1))
y = np.array([rng.random() < taxa[c] for c in X[:, 0]]).astype(int)

ordinal = make_pipeline(OrdinalEncoder(), LogisticRegression())
onehot = make_pipeline(OneHotEncoder(), LogisticRegression())
print("Acurácia com ordinal:", round(cross_val_score(ordinal, X, y, cv=5).mean(), 3))
print("Acurácia com one-hot:", round(cross_val_score(onehot, X, y, cv=5).mean(), 3))
