# Capítulo 5 — solução de referência (PROJ)
import numpy as np
import optuna
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error

optuna.logging.set_verbosity(optuna.logging.WARNING)
X, y = load_diabetes(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
cv = KFold(5, shuffle=True, random_state=0)

print("Linha de base (média) R²:", round(cross_val_score(DummyRegressor(), Xtr, ytr, cv=cv, scoring="r2").mean(), 3))
print("Ridge padrão          R²:", round(cross_val_score(Ridge(), Xtr, ytr, cv=cv, scoring="r2").mean(), 3))

def construir(trial):
    alg = trial.suggest_categorical("algoritmo", ["ridge", "floresta", "boosting", "svr"])
    if alg == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=trial.suggest_float("alpha", 1e-3, 1e3, log=True)))
    if alg == "floresta":
        return RandomForestRegressor(trial.suggest_int("rf_arvores", 20, 150), max_depth=trial.suggest_int("rf_prof", 2, 10),
                                     min_samples_leaf=trial.suggest_int("rf_folha", 1, 15), random_state=0, n_jobs=-1)
    if alg == "boosting":
        return GradientBoostingRegressor(n_estimators=trial.suggest_int("gb_arvores", 20, 200), learning_rate=trial.suggest_float("gb_lr", 0.01, 0.3, log=True),
                                         max_depth=trial.suggest_int("gb_prof", 1, 4), random_state=0)
    return make_pipeline(StandardScaler(), SVR(C=trial.suggest_float("svr_C", 1e-1, 1e3, log=True), epsilon=trial.suggest_float("svr_eps", 1e-2, 1.0, log=True),
                                               gamma=trial.suggest_float("svr_gamma", 1e-4, 1e-1, log=True)))

def objetivo(trial):
    return cross_val_score(construir(trial), Xtr, ytr, cv=cv, scoring="r2").mean()

N = 40; curvas = {"TPE": [], "Aleatória": []}; melhores = []
for semente in range(3):
    for nome, sampler in [("TPE", optuna.samplers.TPESampler(seed=semente)), ("Aleatória", optuna.samplers.RandomSampler(seed=semente))]:
        est = optuna.create_study(direction="maximize", sampler=sampler); est.optimize(objetivo, n_trials=N)
        curvas[nome].append(np.maximum.accumulate([t.value for t in est.trials])); melhores.append((est.best_value, est))
for nome, c in curvas.items():
    c = np.array(c); plt.plot(range(1, N + 1), c.mean(0), label=nome)
    print(f"{nome:10s} melhor R² médio após {N} tentativas: {c[:, -1].mean():.3f} (variação entre sementes: {c[:, -1].min():.3f} a {c[:, -1].max():.3f})")
plt.xlabel("tentativas"); plt.ylabel("melhor R² até agora"); plt.legend(); plt.title("TPE x aleatória (média de 3 sementes)"); plt.show()

valor, est = max(melhores, key=lambda t: t[0])
modelo = construir(optuna.trial.FixedTrial(est.best_params)).fit(Xtr, ytr)
p = modelo.predict(Xte)
print("\nMelhor algoritmo:", est.best_params["algoritmo"], "| R² de validação:", round(valor, 3))
print(f"Teste final: R² = {r2_score(yte, p):.3f} | RMSE = {np.sqrt(mean_squared_error(yte, p)):.1f}")
print(f"Linha de base (média) no teste: RMSE = {np.sqrt(mean_squared_error(yte, DummyRegressor().fit(Xtr, ytr).predict(Xte))):.1f}")
pr = Ridge().fit(Xtr, ytr).predict(Xte)
print(f"Ridge padrão no teste             : R² = {r2_score(yte, pr):.3f} | RMSE = {np.sqrt(mean_squared_error(yte, pr)):.1f}")
