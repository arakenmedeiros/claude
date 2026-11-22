# Capítulo 9 — solução de referência (PROJ)
import numpy as np
import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss

optuna.logging.set_verbosity(optuna.logging.WARNING)
X0, y0 = load_breast_cancer(return_X_y=True)
rng = np.random.default_rng(0)
X = np.hstack([X0, rng.normal(size=(len(y0), 100))])
y = np.where(rng.random(len(y0)) < 0.10, 1 - y0, y0)                      # 10% de rótulos trocados
d = X.shape[1]
Xa, Xte, ya, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
Xtr, Xva, ytr, yva = train_test_split(Xa, ya, test_size=1/3, stratify=ya, random_state=0)
cv = StratifiedKFold(3, shuffle=True, random_state=0)

guardados = {}                                                              # probabilidades de cada tentativa
def objetivo(trial):
    alg = trial.suggest_categorical("alg", ["lr", "svm", "rf"])
    if alg == "lr":
        m = make_pipeline(StandardScaler(), LogisticRegression(C=trial.suggest_float("lr_C", 1e-4, 1e2, log=True), max_iter=3000))
    elif alg == "svm":
        m = make_pipeline(StandardScaler(), SVC(C=trial.suggest_float("svm_C", 1e-2, 1e3, log=True),
                          gamma=trial.suggest_float("svm_gamma", 1e-5, 1e-1, log=True), probability=True, random_state=0))
    else:
        m = RandomForestClassifier(trial.suggest_int("rf_n", 20, 150), max_depth=trial.suggest_int("rf_prof", 2, 14), random_state=0, n_jobs=-1)
    nota = cross_val_score(m, Xtr, ytr, cv=cv).mean()
    m.fit(Xtr, ytr); guardados[trial.number] = (m.predict_proba(Xva), m.predict_proba(Xte))
    return nota

est = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
est.enqueue_trial({"alg": "svm", "svm_C": 1.0, "svm_gamma": 1.0 / d})       # warm start
est.enqueue_trial({"alg": "rf", "rf_n": 100, "rf_prof": 8})
est.optimize(objetivo, n_trials=30)

ids = sorted(guardados); PV = [guardados[i][0] for i in ids]; PT = [guardados[i][1] for i in ids]
melhor = max(est.trials, key=lambda t: t.value).number
print(f"Melhor modelo único (validação cruzada = {est.best_value:.3f}): {est.best_params['alg']}")

escolhidos, atual = [], None                                                # seleção de Caruana na validação
for _ in range(20):
    perdas = []
    for j in range(len(PV)):
        cand = PV[j] if not escolhidos else (atual * len(escolhidos) + PV[j]) / (len(escolhidos) + 1)
        perdas.append(log_loss(yva, np.clip(cand, 1e-12, 1), labels=[0, 1]))
    j = int(np.argmin(perdas)); atual = PV[j] if not escolhidos else (atual * len(escolhidos) + PV[j]) / (len(escolhidos) + 1); escolhidos.append(j)
pesos = np.bincount(escolhidos, minlength=len(PV))

teste_unico = accuracy_score(yte, PT[ids.index(melhor)].argmax(1))
teste_media = accuracy_score(yte, (sum(PT) / len(PT)).argmax(1))
teste_caruana = accuracy_score(yte, (sum(p * w for p, w in zip(PT, pesos)) / pesos.sum()).argmax(1))
print(f"\nTESTE FINAL ({len(yte)} exemplos):")
print(f"  melhor modelo único        : {teste_unico:.3f}")
print(f"  média simples de 30 modelos: {teste_media:.3f}")
print(f"  ensemble de Caruana        : {teste_caruana:.3f} ({(pesos > 0).sum()} modelos distintos)")
