# Capítulo 9 — solução de referência (EX 4)
import numpy as np
import optuna
from sklearn.datasets import load_wine
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)
X, y = load_wine(return_X_y=True); d = X.shape[1]
cv = StratifiedKFold(3, shuffle=True, random_state=0)
PORTFOLIO = [(1, 1), (10, 1), (10, 0.1)]                                  # (C, gamma relativo)

def objetivo(trial):
    C = trial.suggest_float("C", 1e-2, 1e3, log=True)
    g_rel = trial.suggest_float("gamma_rel", 1e-3, 1e1, log=True)
    return cross_val_score(make_pipeline(StandardScaler(), SVC(C=C, gamma=g_rel / d)), X, y, cv=cv).mean()

def rodar(semente, quente):
    est = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=semente))
    if quente:
        for C, g in PORTFOLIO: est.enqueue_trial({"C": C, "gamma_rel": g})
    est.optimize(objetivo, n_trials=10)
    return np.maximum.accumulate([t.value for t in est.trials])

frio = np.array([rodar(s, False) for s in range(5)]); quente = np.array([rodar(s, True) for s in range(5)])
print("tentativas | partida a frio | com warm start")
for k in (1, 3, 10):
    print(f"{k:10d} | {frio[:, k-1].mean():14.3f} | {quente[:, k-1].mean():.3f}")
