# Capítulo 1 — Bem-vindo ao AutoML
# Bloco 4. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

X, y = load_wine(return_X_y=True)
rng = np.random.default_rng(0)

def sortear_candidato():
    """Sorteia um ponto do espaço CASH: algoritmo + hiperparâmetros."""
    alg = rng.choice(["arvore", "knn", "logistica", "floresta"])
    if alg == "arvore":
        hp = {"max_depth": int(rng.integers(1, 15)), "min_samples_leaf": int(rng.integers(1, 20))}
        modelo = DecisionTreeClassifier(random_state=0, **hp)
    elif alg == "knn":
        hp = {"n_neighbors": int(rng.integers(1, 30)), "weights": str(rng.choice(["uniform", "distance"]))}
        modelo = make_pipeline(StandardScaler(), KNeighborsClassifier(**hp))
    elif alg == "logistica":
        hp = {"C": float(10 ** rng.uniform(-3, 3))}
        modelo = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, **hp))
    else:
        hp = {"n_estimators": int(rng.integers(10, 200)), "max_depth": int(rng.integers(2, 15))}
        modelo = RandomForestClassifier(random_state=0, **hp)
    return alg, hp, modelo

melhor = (-1, None, None)
for tentativa in range(40):
    alg, hp, modelo = sortear_candidato()
    nota = float(cross_val_score(modelo, X, y, cv=5).mean())
    if nota > melhor[0] + 1e-9:
        melhor = (nota, str(alg), hp)
        print(f"tentativa {tentativa:2d}: novo melhor = {nota:.3f}  {alg}")

nota, alg, hp = melhor
print(f"\nMelhor encontrado: {alg}, acurácia = {nota:.3f}")
print("Hiperparâmetros:", {k: round(v, 3) if isinstance(v, float) else v for k, v in hp.items()})
