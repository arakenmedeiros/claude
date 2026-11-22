# Capítulo 10 — Ferramentas de AutoML na prática
# Bloco 2. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
import warnings; warnings.filterwarnings("ignore")
from flaml import AutoML
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

a = AutoML()
a.fit(Xtr, ytr, task="classification", time_budget=8, metric="accuracy", eval_method="cv",
      n_splits=3, seed=0, verbose=0, estimator_list=["lgbm", "rf", "extra_tree", "lrl1"])

print("Vencedor:", a.best_estimator)
print("Configuração:")
for k, v in a.best_config.items(): print(f"  {k:<14s} {v if isinstance(v, str) else round(float(v), 3)}")
print("Modelo pronto:", type(a.model.estimator).__name__)
print()
print("Melhor erro por algoritmo (validação cruzada):")
for nome, erro in sorted(a.best_loss_per_estimator.items(), key=lambda t: t[1]):
    print(f"  {nome:<11s} erro = {erro:.4f}")
