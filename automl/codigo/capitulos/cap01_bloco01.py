# Capítulo 1 — Bem-vindo ao AutoML
# Bloco 1. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
import sys
from importlib.metadata import version
print("python      :", sys.version.split()[0])
for pacote in ["numpy", "pandas", "scipy", "matplotlib", "scikit-learn", "optuna", "flaml", "shap", "lightgbm", "xgboost", "statsmodels", "joblib"]:
    print(f"{pacote:<12s}:", version(pacote))
