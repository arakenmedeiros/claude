# Capítulo 5 — Otimização bayesiana
# Bloco 3. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
# --- blocos anteriores deste capítulo (encadeados) ---
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
# --- fim dos blocos anteriores ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def objetivo_cash(trial):
    algoritmo = trial.suggest_categorical("algoritmo", ["svm", "floresta", "logistica"])
    if algoritmo == "svm":
        modelo = SVC(C=trial.suggest_float("svm_C", 1e-3, 1e4, log=True),
                     gamma=trial.suggest_float("svm_gamma", 1e-8, 1e1, log=True))
    elif algoritmo == "floresta":
        modelo = RandomForestClassifier(n_estimators=trial.suggest_int("rf_arvores", 20, 150),
                                        max_depth=trial.suggest_int("rf_profundidade", 3, 20),
                                        random_state=0, n_jobs=-1)
    else:
        modelo = make_pipeline(StandardScaler(), LogisticRegression(
                     C=trial.suggest_float("lr_C", 1e-4, 1e3, log=True), max_iter=2000))
    return cross_val_score(modelo, X, y, cv=cv).mean()

cash = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
cash.optimize(objetivo_cash, n_trials=30)

print("Algoritmo vencedor:", cash.best_params["algoritmo"], "| acurácia =", round(cash.best_value, 4))
resumo = {}
for t in cash.trials:
    r = resumo.setdefault(t.params["algoritmo"], {"tentativas": 0, "melhor": 0.0})
    r["tentativas"] += 1
    r["melhor"] = max(r["melhor"], t.value)
for nome, r in resumo.items():
    print(f"{nome:10s} tentativas = {r['tentativas']:2d} | melhor nota = {r['melhor']:.3f}")
