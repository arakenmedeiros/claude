# Capítulo 4 — Busca em grade e aleatória
# Bloco 6. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, randint, uniform
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

X, y = load_digits(return_X_y=True)
espaco = {"n_estimators": randint(10, 200), "max_depth": randint(2, 20),
          "max_features": uniform(0.05, 0.9), "min_samples_leaf": randint(1, 15)}
busca = RandomizedSearchCV(RandomForestClassifier(random_state=0, n_jobs=-1), espaco, n_iter=40, cv=3, random_state=0).fit(X, y)

res = pd.DataFrame(busca.cv_results_)
print("Correlação de Spearman entre cada hiperparâmetro e a acurácia:")
for h in espaco:
    rho = spearmanr(res["param_" + h].astype(float), res["mean_test_score"])[0]
    print(f"  {h:17s} rho = {rho:+.2f}")
print(f"Acurácia: mínima = {res['mean_test_score'].min():.3f} | máxima = {res['mean_test_score'].max():.3f}")
