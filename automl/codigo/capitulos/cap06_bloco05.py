# Capítulo 6 — Métodos multifidelidade: Successive Halving, Hyperband e BOHB
# Bloco 5. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
import numpy as np
from sklearn.experimental import enable_halving_search_cv    # habilita as classes experimentais
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV, StratifiedKFold
from sklearn.datasets import load_digits
from sklearn.svm import SVC

X, y = load_digits(return_X_y=True); X = X / 16.0
cv = StratifiedKFold(5, shuffle=True, random_state=0)
grade = {"C": np.logspace(-1, 3, 6), "gamma": np.logspace(-5, -1, 6)}      # 36 combinações

completa = GridSearchCV(SVC(), grade, cv=cv).fit(X, y)
halving = HalvingGridSearchCV(SVC(), grade, cv=cv, factor=3, random_state=0).fit(X, y)

n_treinos = len(completa.cv_results_["params"]) * 5
custo_completa = n_treinos * int(len(X) * 0.8)
custo_halving = sum(c * r * 5 * 0.8 for c, r in zip(halving.n_candidates_, halving.n_resources_))

print(f"Grade completa : {n_treinos} treinos | melhor acurácia = {completa.best_score_:.3f}")
print(f"Halving        : candidatos por rodada {halving.n_candidates_} | exemplos por rodada {halving.n_resources_}")
print(f"                 melhor acurácia (na última rodada) = {halving.best_score_:.3f}")
print(f"Custo aproximado (exemplos processados): grade = {custo_completa:,.0f} | halving = {custo_halving:,.0f}")
print(f"Economia: {100 * (1 - custo_halving / custo_completa):.0f}%")
