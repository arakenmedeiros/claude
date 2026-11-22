# Capítulo 1 — Bem-vindo ao AutoML
# Bloco 2. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
from sklearn.datasets import load_wine
import numpy as np

dados = load_wine()
X, y = dados.data, dados.target
print("Formato de X:", X.shape, "(linhas = vinhos, colunas = medidas)")
print("Medidas:", list(dados.feature_names)[:4], "...")
print("Vinhos por categoria:", np.bincount(y))
