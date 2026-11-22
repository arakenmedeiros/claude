# Capítulo 1 — solução de referência (EX 2)
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

X, y = load_wine(return_X_y=True)
rng = np.random.default_rng(0)

def sortear():
    alg = rng.choice(["arvore", "knn", "logistica", "floresta", "svm", "gboost"])
    if alg == "arvore":
        hp = {"max_depth": int(rng.integers(1, 15)), "min_samples_leaf": int(rng.integers(1, 20))}
        m = DecisionTreeClassifier(random_state=0, **hp)
    elif alg == "knn":
        hp = {"n_neighbors": int(rng.integers(1, 30))}
        m = make_pipeline(StandardScaler(), KNeighborsClassifier(**hp))
    elif alg == "logistica":
        hp = {"C": float(10 ** rng.uniform(-3, 3))}
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, **hp))
    elif alg == "floresta":
        hp = {"n_estimators": int(rng.integers(10, 150)), "max_depth": int(rng.integers(2, 15))}
        m = RandomForestClassifier(random_state=0, **hp)
    elif alg == "svm":
        hp = {"C": float(10 ** rng.uniform(-2, 3)), "gamma": float(10 ** rng.uniform(-4, 0))}
        m = make_pipeline(StandardScaler(), SVC(**hp))
    else:
        hp = {"n_estimators": int(rng.integers(20, 150)), "learning_rate": float(10 ** rng.uniform(-2, 0))}
        m = GradientBoostingClassifier(random_state=0, **hp)
    return str(alg), hp, m

resultados = []
for _ in range(60):
    alg, hp, m = sortear()
    resultados.append((float(cross_val_score(m, X, y, cv=5).mean()), alg, hp))

resultados.sort(key=lambda t: -t[0])
for nota, alg, hp in resultados[:3]:
    print(f"{nota:.3f}  {alg:10s}", {k: round(v, 3) if isinstance(v, float) else v for k, v in hp.items()})
print("Algoritmos entre os 10 melhores:", sorted({a for _, a, _ in resultados[:10]}))
