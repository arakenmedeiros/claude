# Capítulo 4 — solução de referência (PROJ)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import loguniform, randint, uniform, spearmanr
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold

X, y = load_digits(return_X_y=True); X = X / 16.0
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
cv = StratifiedKFold(3, shuffle=True, random_state=0)

algoritmos = {
    "SVM": (SVC(), {"C": loguniform(1e-2, 1e3), "gamma": loguniform(1e-5, 1)}),
    "Floresta": (RandomForestClassifier(random_state=0, n_jobs=-1),
                 {"n_estimators": randint(20, 150), "max_depth": randint(3, 20), "max_features": uniform(0.05, 0.9)}),
    "Boosting": (HistGradientBoostingClassifier(max_iter=60, random_state=0),
                 {"learning_rate": loguniform(0.02, 0.5), "max_leaf_nodes": randint(4, 40), "l2_regularization": loguniform(1e-4, 10)}),
}
buscas = {}
for nome, (modelo, espaco) in algoritmos.items():
    buscas[nome] = RandomizedSearchCV(modelo, espaco, n_iter=12, cv=cv, random_state=0, refit=True).fit(Xtr, ytr)
    print(f"{nome:9s} melhor validação = {buscas[nome].best_score_:.3f}")

for nome, b in buscas.items():
    notas = pd.Series(b.cv_results_["mean_test_score"])
    plt.plot(range(1, 13), notas.cummax(), "o-", label=nome)
plt.xlabel("tentativas"); plt.ylabel("melhor nota até agora"); plt.legend(); plt.title("Convergência da busca"); plt.show()

vencedor = max(buscas, key=lambda n: buscas[n].best_score_)
res = pd.DataFrame(buscas[vencedor].cv_results_)
print("\nVencedor:", vencedor)
for h in algoritmos[vencedor][1]:
    print(f"  {h:18s} rho = {spearmanr(res['param_' + h].astype(float), res['mean_test_score'])[0]:+.2f}")
print(f"Teste final do vencedor: {buscas[vencedor].score(Xte, yte):.3f}")
