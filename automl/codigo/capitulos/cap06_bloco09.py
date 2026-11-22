# Capítulo 6 — Métodos multifidelidade: Successive Halving, Hyperband e BOHB
# Bloco 9. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.datasets import load_digits
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = load_digits(return_X_y=True)
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
esc = StandardScaler().fit(Xtr); Xtr, Xva = esc.transform(Xtr), esc.transform(Xva)

EPOCAS = 60
def objetivo(trial):
    modelo = SGDClassifier(loss=trial.suggest_categorical("loss", ["hinge", "log_loss", "modified_huber"]),
                           alpha=trial.suggest_float("alpha", 1e-8, 1e1, log=True),
                           eta0=trial.suggest_float("eta0", 1e-6, 1e1, log=True),
                           learning_rate="constant", random_state=0)
    for epoca in range(EPOCAS):
        modelo.partial_fit(Xtr, ytr, classes=np.arange(10))
        nota = modelo.score(Xva, yva)
        trial.report(nota, epoca)                      # nota parcial
        if trial.should_prune():                       # o podador manda parar?
            raise optuna.TrialPruned()
    return nota

for nome, podador in [("sem poda", optuna.pruners.NopPruner()), ("MedianPruner", optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3))]:
    estudo = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0), pruner=podador)
    estudo.optimize(objetivo, n_trials=40)
    podadas = sum(t.state == optuna.trial.TrialState.PRUNED for t in estudo.trials)
    epocas = sum(len(t.intermediate_values) for t in estudo.trials)
    print(f"{nome:13s}: tentativas podadas = {podadas:2d}/40 | épocas treinadas = {epocas:4d} | melhor acurácia = {estudo.best_value:.3f}")
