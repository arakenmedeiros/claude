# Capítulo 5 — Otimização bayesiana
# Bloco 2. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
import optuna
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)      # silencia mensagens de progresso

X, y = load_digits(return_X_y=True)
X = X / 16.0
cv = StratifiedKFold(3, shuffle=True, random_state=0)

def objetivo(trial):
    # O espaço de busca é descrito aqui dentro, com código comum
    C = trial.suggest_float("C", 1e-3, 1e4, log=True)
    gamma = trial.suggest_float("gamma", 1e-8, 1e1, log=True)
    return cross_val_score(SVC(C=C, gamma=gamma), X, y, cv=cv).mean()

estudo = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
estudo.optimize(objetivo, n_trials=30)

print("Melhor acurácia:", round(estudo.best_value, 4))
print("Melhores valores: C = %.2f, gamma = %.5f" % (estudo.best_params["C"], estudo.best_params["gamma"]))
print("Tentativa em que o melhor apareceu:", estudo.best_trial.number)
