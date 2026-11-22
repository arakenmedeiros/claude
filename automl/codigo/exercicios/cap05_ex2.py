# Capítulo 5 — solução de referência (EX 2)
import numpy as np
import optuna
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)
X, y = load_breast_cancer(return_X_y=True)
cv = StratifiedKFold(5, shuffle=True, random_state=0)

def objetivo(trial):
    modelo = RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 20, 150),
        max_depth=trial.suggest_int("max_depth", 2, 15),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        max_features=trial.suggest_float("max_features", 0.1, 1.0),
        random_state=0, n_jobs=-1)
    return cross_val_score(modelo, X, y, cv=cv).mean()

estudo = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
estudo.optimize(objetivo, n_trials=25)
print("Melhor nota:", round(estudo.best_value, 4))
print("Melhores valores:", {k: round(v, 3) if isinstance(v, float) else v for k, v in estudo.best_params.items()})

notas = [t.value for t in estudo.trials]
plt.plot(range(1, 26), np.maximum.accumulate(notas), "o-"); plt.xlabel("tentativa"); plt.ylabel("melhor nota até agora"); plt.show()

imp = optuna.importance.get_param_importances(estudo)
print("Importância dos hiperparâmetros:")
for nome, v in imp.items():
    print(f"  {nome:17s} {v:.2f}")
print("Variação das notas: de", round(min(notas), 3), "a", round(max(notas), 3))
