# Capítulo 4 — solução de referência (EX 1)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

X, y = load_wine(return_X_y=True)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
pipe = make_pipeline(StandardScaler(), KNeighborsClassifier())
grade = {"kneighborsclassifier__n_neighbors": list(range(1, 26, 2)),
         "kneighborsclassifier__weights": ["uniform", "distance"]}
busca = GridSearchCV(pipe, grade, cv=cv).fit(X, y)

n = len(busca.cv_results_["params"])
print("Combinações:", n, "| treinos:", n * 5)
print("Melhor:", busca.best_params_, "| acurácia =", round(busca.best_score_, 3))

res = pd.DataFrame(busca.cv_results_)
res["k"] = res["param_kneighborsclassifier__n_neighbors"].astype(int)
res["peso"] = res["param_kneighborsclassifier__weights"].astype(str)
for peso, g in res.groupby("peso"):
    plt.plot(g["k"], g["mean_test_score"], "o-", label=peso)
    print(f"peso {peso:8s}: melhor = {g['mean_test_score'].max():.3f} | pior = {g['mean_test_score'].min():.3f}")
plt.xlabel("n_neighbors"); plt.ylabel("acurácia (validação cruzada)"); plt.legend(); plt.show()
