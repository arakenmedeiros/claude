# Capítulo 6 — solução de referência (EX 2)
import numpy as np
from scipy.stats import randint, uniform
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import RandomizedSearchCV, HalvingRandomSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier

X, y = load_digits(return_X_y=True)
espaco = {"max_depth": randint(3, 25), "max_features": uniform(0.05, 0.55), "min_samples_leaf": randint(1, 8)}

aleat = RandomizedSearchCV(RandomForestClassifier(n_estimators=150, random_state=0, n_jobs=-1), espaco, n_iter=30, cv=3, random_state=0).fit(X, y)
halv = HalvingRandomSearchCV(RandomForestClassifier(random_state=0, n_jobs=-1), espaco, n_candidates=30, resource="n_estimators",
                             max_resources=150, min_resources=10, factor=3, cv=3, random_state=0).fit(X, y)

custo_aleat = 30 * 150 * 3
custo_halv = sum(c * r * 3 for c, r in zip(halv.n_candidates_, halv.n_resources_))
print("Halving: candidatos por rodada", halv.n_candidates_, "| árvores por rodada", halv.n_resources_)
print(f"Aleatória : melhor = {aleat.best_score_:.3f} | custo = {custo_aleat:,} árvores treinadas")
print(f"Halving   : melhor = {halv.best_score_:.3f} | custo = {custo_halv:,} árvores treinadas")
print(f"Economia de custo: {100 * (1 - custo_halv / custo_aleat):.0f}%")
