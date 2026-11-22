# Capítulo 14 — solução de referência (EX 5)
import warnings; warnings.filterwarnings("ignore")
import pickle, time
import numpy as np, optuna
from sklearn.datasets import load_digits
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
optuna.logging.set_verbosity(optuna.logging.WARNING)

X, y = load_digits(return_X_y=True)
def objetivo(t):
    alg = t.suggest_categorical("alg", ["lr", "rf", "hgb"])
    if alg == "lr": m = make_pipeline(StandardScaler(), LogisticRegression(C=t.suggest_float("C", 1e-3, 1e2, log=True), max_iter=2000))
    elif alg == "rf": m = RandomForestClassifier(t.suggest_int("n", 20, 300), max_depth=t.suggest_int("prof", 3, 16), random_state=0, n_jobs=1)
    else: m = HistGradientBoostingClassifier(max_iter=t.suggest_int("it", 20, 200), random_state=0)
    acc = cross_val_score(m, X, y, cv=3).mean(); m.fit(X, y); m.predict(X[:1]); ts = []
    for _ in range(20):
        t0 = time.perf_counter(); m.predict(X[:1]); ts.append(time.perf_counter() - t0)
    lat, kb = np.median(ts) * 1000, len(pickle.dumps(m)) / 1024
    t.set_user_attr("m", (lat, kb)); t.set_user_attr("c", (lat - 2.0, kb - 2000.0)); return acc
est = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0, constraints_func=lambda t: t.user_attrs["c"]))
est.optimize(objetivo, n_trials=30)
viavel = lambda t: all(c <= 0 for c in t.user_attrs["c"]); V = [t for t in est.trials if viavel(t)]
print(f"Tentativas viáveis: {len(V)} de 30")
for rot, t in (("melhor de todas", max(est.trials, key=lambda t: t.value)), ("melhor viável", max(V, key=lambda t: t.value) if V else None)):
    if t is None: print(rot, ": nenhuma"); continue
    print(f"{rot:<16s} {t.params['alg']:<4s} acc = {t.value:.3f} | {t.user_attrs['m'][0]:.2f} ms | {t.user_attrs['m'][1]:.0f} kB | viável: {viavel(t)}")
