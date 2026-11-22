# Capítulo 2 — Aprendizado de máquina em um capítulo
# Bloco 8. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

rng = np.random.default_rng(0)
# Dados SEM nenhum sinal: o alvo é puro sorteio
X = rng.normal(size=(120, 15)); y = rng.integers(0, 2, 120)
X_novo = rng.normal(size=(5000, 15)); y_novo = rng.integers(0, 2, 5000)

melhor_nota, melhor_modelo = 0, None
for _ in range(300):                       # 300 candidatos aleatórios
    if rng.random() < 0.5:
        m = KNeighborsClassifier(n_neighbors=int(rng.integers(1, 40)))
    else:
        m = DecisionTreeClassifier(max_depth=int(rng.integers(1, 10)), random_state=0)
    nota = cross_val_score(m, X, y, cv=5).mean()
    if nota > melhor_nota:
        melhor_nota, melhor_modelo = nota, m

melhor_modelo.fit(X, y)
print(f"Melhor nota de validação entre 300 modelos: {melhor_nota:.3f}")
print(f"Acurácia do mesmo modelo em dados NOVOS     : {melhor_modelo.score(X_novo, y_novo):.3f}")
