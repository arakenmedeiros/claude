# Capítulo 14 — MLOps, implantação e projeto integrador
# Bloco 4. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
import warnings; warnings.filterwarnings("ignore")
import pickle, time
import numpy as np, optuna
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
optuna.logging.set_verbosity(optuna.logging.WARNING)

def gera(n, seed, vies=0.6):
    r = np.random.default_rng(seed)
    g = (r.random(n) < 0.3).astype(int); z = r.normal(0, 1, n)
    renda = np.exp(0.5 * z + 0.35 * r.normal(size=n) - 0.25 * g) * 3.0
    emprego = np.clip(5 + 2 * z + r.normal(0, 2, n), 0, None); divida = np.clip(0.4 - 0.1 * z + r.normal(0, 0.1, n), 0, 1)
    score = z + r.normal(0, 1, n); y = ((z - vies * g + r.normal(0, 0.5, n)) > 0).astype(int)
    return np.c_[renda, emprego, divida, score], g, y

X, g, y = gera(6000, seed=3)
Xa, Xte, ya, yte, ga, gte = train_test_split(X, y, g, test_size=0.3, random_state=0, stratify=y)
Xtr, Xva, ytr, yva, gtr, gva = train_test_split(Xa, ya, ga, test_size=0.25, random_state=1, stratify=ya)
LIMITES = {"latência (ms)": 3.0, "tamanho (kB)": 1000.0, "disparidade": 0.065}

def constroi(alg, p):
    if alg == "lr": return make_pipeline(StandardScaler(), LogisticRegression(C=p["C"], max_iter=2000))
    if alg == "rf": return RandomForestClassifier(p["n"], max_depth=p["prof"], random_state=0, n_jobs=1)
    return lgb.LGBMClassifier(n_estimators=p["n"], learning_rate=p["lr"], num_leaves=p["folhas"], random_state=0, verbose=-1, n_jobs=1)

def objetivo(t):
    alg = t.suggest_categorical("alg", ["lr", "rf", "lgbm"])
    if alg == "lr": p = {"C": t.suggest_float("C", 1e-3, 1e2, log=True)}
    elif alg == "rf": p = {"n": t.suggest_int("n_rf", 20, 400), "prof": t.suggest_int("prof", 3, 16)}
    else: p = {"n": t.suggest_int("n_lgb", 20, 400), "lr": t.suggest_float("lr", 0.01, 0.3, log=True), "folhas": t.suggest_int("folhas", 4, 64)}
    m = constroi(alg, p); acc = cross_val_score(m, Xtr, ytr, cv=3).mean(); m.fit(Xtr, ytr)
    um = Xva[:1]; m.predict(um); tempos = []
    for _ in range(20):
        t0 = time.perf_counter(); m.predict(um); tempos.append(time.perf_counter() - t0)
    lat, kb = np.median(tempos) * 1000, len(pickle.dumps(m)) / 1024
    pr = m.predict(Xva); disp = abs(pr[gva == 0].mean() - pr[gva == 1].mean())
    t.set_user_attr("medidas", (lat, kb, disp))
    t.set_user_attr("c", (lat - 3.0, kb - 1000.0, (disp - 0.065) * 100))       # restrições: valor <= 0 significa viável
    return acc

viavel = lambda t: all(c <= 0 for c in t.user_attrs["c"])
for nome, amostrador in (("TPE comum", optuna.samplers.TPESampler(seed=0)),
                         ("TPE com restrições", optuna.samplers.TPESampler(seed=0, constraints_func=lambda t: t.user_attrs["c"]))):
    estudo = optuna.create_study(direction="maximize", sampler=amostrador); estudo.optimize(objetivo, n_trials=60)
    V = [t for t in estudo.trials if viavel(t)]; geral = max(estudo.trials, key=lambda t: t.value)
    print(f"{nome}: tentativas viáveis = {len(V)} de 60")
    for rot, t in (("melhor de todas", geral), ("melhor VIÁVEL", max(V, key=lambda t: t.value) if V else None)):
        if t is None: print(f"   {rot}: nenhuma"); continue
        lat, kb, disp = t.user_attrs["medidas"]
        print(f"   {rot:<15s}: {t.params['alg']:<4s} acc (cv) = {t.value:.3f} | {lat:5.2f} ms | {kb:7.1f} kB | disparidade = {disp:.3f} | viável: {'sim' if viavel(t) else 'não'}")
