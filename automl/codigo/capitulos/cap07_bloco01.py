# Capítulo 7 — Métodos evolutivos e programação genética
# Bloco 1. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
import copy
import numpy as np
from sklearn.datasets import load_digits
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Dados: 300 imagens de Digits + 200 colunas de ruído
X0, y0 = load_digits(return_X_y=True)
X0 = X0 / 16.0
rng_dados = np.random.default_rng(0)
idx = rng_dados.choice(len(y0), 300, replace=False)
X = np.hstack([X0[idx], rng_dados.normal(size=(300, 200))])
y = y0[idx]
print("Formato dos dados:", X.shape, "(64 colunas úteis + 200 de ruído)")

def sortear_modelo(rng, modelo):
    """Sorteia os hiperparâmetros do modelo escolhido."""
    if modelo == "knn":       return {"k": int(rng.integers(1, 30))}
    if modelo == "svm":       return {"C": float(10 ** rng.uniform(-2, 3)), "gamma": float(10 ** rng.uniform(-5, 0))}
    if modelo == "logistica": return {"C": float(10 ** rng.uniform(-3, 3))}
    return {"arvores": int(rng.integers(20, 150)), "prof": int(rng.integers(3, 20))}

def sortear_individuo(rng):
    g = {"escala": str(rng.choice(["nenhuma", "padrao", "minmax"])),
         "pca": int(rng.choice([0, 0, 10, 20, 30, 40])),
         "modelo": str(rng.choice(["knn", "svm", "logistica", "floresta"]))}
    g.update(sortear_modelo(rng, g["modelo"]))
    return g
