# Capítulo 5 — solução de referência (EX 3)
import optuna
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)
X, y = load_digits(return_X_y=True); X = X / 16.0
cv = StratifiedKFold(3, shuffle=True, random_state=0)

def rodar(gamma_min, gamma_max, semente=0, n=30):
    def objetivo(trial):
        alg = trial.suggest_categorical("algoritmo", ["svm", "floresta", "logistica"])
        if alg == "svm":
            m = SVC(C=trial.suggest_float("C", 1e-3, 1e4, log=True), gamma=trial.suggest_float("gamma", gamma_min, gamma_max, log=True))
        elif alg == "floresta":
            m = RandomForestClassifier(trial.suggest_int("arvores", 20, 150), max_depth=trial.suggest_int("prof", 3, 20), random_state=0, n_jobs=-1)
        else:
            m = make_pipeline(StandardScaler(), LogisticRegression(C=trial.suggest_float("lrC", 1e-4, 1e3, log=True), max_iter=2000))
        return cross_val_score(m, X, y, cv=cv).mean()
    est = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=semente))
    est.optimize(objetivo, n_trials=n)
    resumo = {}
    for t in est.trials:
        r = resumo.setdefault(t.params["algoritmo"], [0, 0.0]); r[0] += 1; r[1] = max(r[1], t.value)
    return est.best_params["algoritmo"], est.best_value, resumo

for rotulo, (a, b) in [("gamma 1e-8 a 10 ", (1e-8, 10)), ("gamma 1e-5 a 1  ", (1e-5, 1))]:
    alg, nota, resumo = rodar(a, b)
    print(f"{rotulo}: vencedor = {alg} ({nota:.3f})")
    for nome, (n, m) in resumo.items():
        print(f"    {nome:10s} tentativas = {n:2d} | melhor = {m:.3f}")
