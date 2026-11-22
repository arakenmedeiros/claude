# Capítulo 3 — Pipelines e pré-processamento
# Bloco 1. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine

rng = np.random.default_rng(0)
df = load_wine(as_frame=True).frame.copy()

# Coluna categórica simulada ("safra"), levemente ligada à classe do vinho
safras = np.array(["2018", "2019", "2020"])
df["safra"] = np.where(rng.random(len(df)) < 0.8, safras[df["target"].values % 3], rng.choice(safras, len(df)))

# 8% de valores ausentes, ao acaso, em quatro colunas
for coluna in ["alcohol", "malic_acid", "flavanoids", "proline"]:
    df.loc[rng.random(len(df)) < 0.08, coluna] = np.nan

y = df.pop("target")
X = df
print("Formato de X:", X.shape)
print("\nValores ausentes por coluna:")
print(X.isna().sum()[X.isna().sum() > 0].to_string())
print("\nColunas numéricas:", X.select_dtypes("number").shape[1], "| colunas de texto:", X.select_dtypes(exclude="number").shape[1])
