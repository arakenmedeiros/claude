# Capítulo 6 — solução de referência (PROJ)
import numpy as np
from scipy.stats import loguniform, randint
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, RandomizedSearchCV, HalvingRandomSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import HistGradientBoostingClassifier

X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
espaco = {"learning_rate": loguniform(0.02, 0.5), "max_leaf_nodes": randint(4, 40),
          "l2_regularization": loguniform(1e-4, 10), "min_samples_leaf": randint(5, 40)}

aleat = RandomizedSearchCV(HistGradientBoostingClassifier(max_iter=100, random_state=0), espaco, n_iter=24, cv=3, random_state=0, n_jobs=-1).fit(Xtr, ytr)
halv = HalvingRandomSearchCV(HistGradientBoostingClassifier(random_state=0), espaco, n_candidates=24, resource="max_iter",
                             min_resources=10, max_resources=100, factor=3, cv=3, random_state=0, n_jobs=-1).fit(Xtr, ytr)

custo_a = 24 * 100 * 3
custo_h = sum(c * r * 3 for c, r in zip(halv.n_candidates_, halv.n_resources_))
print("Halving: candidatos por rodada", halv.n_candidates_, "| iterações por rodada", halv.n_resources_)
print(f"Custo aproximado (iterações de boosting): aleatória = {custo_a:,} | halving = {custo_h:,} | economia = {100 * (1 - custo_h / custo_a):.0f}%")
print(f"Validação: aleatória = {aleat.best_score_:.3f} | halving = {halv.best_score_:.3f}")
print(f"Teste final: aleatória = {aleat.score(Xte, yte):.3f} | halving = {halv.score(Xte, yte):.3f}")
