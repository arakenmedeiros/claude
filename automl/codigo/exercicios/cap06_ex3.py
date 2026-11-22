# Capítulo 6 — solução de referência (EX 3)
import numpy as np
import optuna
from sklearn.datasets import load_digits
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

optuna.logging.set_verbosity(optuna.logging.WARNING)
X, y = load_digits(return_X_y=True)
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
esc = StandardScaler().fit(Xtr); Xtr, Xva = esc.transform(Xtr), esc.transform(Xva)
EPOCAS = 60

def objetivo(trial):
    m = SGDClassifier(loss=trial.suggest_categorical("loss", ["hinge", "log_loss", "modified_huber"]),
                      alpha=trial.suggest_float("alpha", 1e-8, 1e1, log=True), eta0=trial.suggest_float("eta0", 1e-6, 1e1, log=True),
                      learning_rate="constant", random_state=0)
    for epoca in range(EPOCAS):
        m.partial_fit(Xtr, ytr, classes=np.arange(10))
        nota = m.score(Xva, yva)
        trial.report(nota, epoca)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return nota

podadores = {"sem poda": optuna.pruners.NopPruner(),
             "MedianPruner": optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
             "HyperbandPruner": optuna.pruners.HyperbandPruner(min_resource=3, max_resource=EPOCAS, reduction_factor=3)}
for nome, p in podadores.items():
    est = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0), pruner=p)
    est.optimize(objetivo, n_trials=40)
    podadas = sum(t.state == optuna.trial.TrialState.PRUNED for t in est.trials)
    epocas = sum(len(t.intermediate_values) for t in est.trials)
    print(f"{nome:16s} podadas = {podadas:2d}/40 | épocas = {epocas:4d} | melhor acurácia = {est.best_value:.3f}")
