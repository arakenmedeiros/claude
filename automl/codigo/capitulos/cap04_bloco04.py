# Capítulo 4 — Busca em grade e aleatória
# Bloco 4. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
import numpy as np
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold

X, y = load_digits(return_X_y=True); X = X / 16.0
cv = StratifiedKFold(5, shuffle=True, random_state=0)
nota = lambda C, g, tol: cross_val_score(SVC(C=C, gamma=g, tol=tol), X, y, cv=cv).mean()

# Grade: 3 valores de cada botão = 27 combinações
melhor_grade = max(nota(C, g, t) for C in [0.1, 10, 1000] for g in [1e-4, 1e-3, 1e-2] for t in [1e-4, 1e-3, 1e-2])

# Aleatória: 27 sorteios, repetida com 8 sementes
resultados = []
for semente in range(8):
    rng = np.random.default_rng(semente)
    resultados.append(max(nota(10 ** rng.uniform(-2, 3), 10 ** rng.uniform(-5, 0), 10 ** rng.uniform(-4, -2)) for _ in range(27)))

print(f"Grade (27 tentativas)         : melhor = {melhor_grade:.3f}")
print(f"Aleatória (27 tentativas) x 8 : melhor = {np.mean(resultados):.3f} em média | pior das 8 = {min(resultados):.3f}")
