# Capítulo 14 — solução de referência (PROJ)
import warnings; warnings.filterwarnings("ignore")
import hashlib, json, os, sys, tempfile, time
import joblib, numpy as np, optuna, sklearn
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
optuna.logging.set_verbosity(optuna.logging.WARNING)

def gera(n=6000, vies=0.6, p_min=0.3, seed=0, infl=0.0):
    r = np.random.default_rng(seed)
    g = (r.random(n) < p_min).astype(int); z = r.normal(0, 1, n)
    renda = np.exp(0.5 * z + 0.35 * r.normal(size=n) - 0.25 * g) * 3.0 * (1 + infl)
    emprego = np.clip(5 + 2 * z + r.normal(0, 2, n), 0, None)
    divida = np.clip(0.4 - 0.1 * z + r.normal(0, 0.1, n), 0, 1)
    cep = 0.8 * g + 0.6 * r.normal(size=n); score = z + r.normal(0, 1, n)
    y_real = (z > 0).astype(int); y_hist = ((z - vies * g + r.normal(0, 0.5, n)) > 0).astype(int)
    return np.c_[renda, emprego, divida, cep, score], g, y_hist, y_real
def psi(ref, novo, k=10):
    cortes = np.quantile(ref, np.linspace(0, 1, k + 1)[1:-1])
    a = np.bincount(np.searchsorted(cortes, ref), minlength=k) / len(ref)
    b = np.bincount(np.searchsorted(cortes, novo), minlength=k) / len(novo)
    a, b = np.clip(a, 1e-4, None), np.clip(b, 1e-4, None)
    return float(np.sum((b - a) * np.log(b / a)))

NOMES = ["renda", "tempo_emprego", "razao_divida", "score_historico"]
X, g, yh, yr = gera(6000, vies=0.5, p_min=0.25, seed=11); X = X[:, [0, 1, 2, 4]]            # sem cep e sem grupo
Xa, Xte, ya, yte, ga, gte, yra, yrte = train_test_split(X, yh, g, yr, test_size=0.25, random_state=0, stratify=yh)
Xtr, Xv, ytr, yv, gtr, gv = train_test_split(Xa, ya, ga, test_size=0.25, random_state=0, stratify=ya)

def constroi(p):
    if p["alg"] == "lr": return make_pipeline(StandardScaler(), LogisticRegression(C=p["C"], max_iter=2000))
    if p["alg"] == "rf": return RandomForestClassifier(p["n"], max_depth=p["prof"], random_state=0, n_jobs=-1)
    return lgb.LGBMClassifier(n_estimators=p["n"], learning_rate=p["lr"], num_leaves=p["folhas"], random_state=0, verbose=-1)

def objetivo(t):
    alg = t.suggest_categorical("alg", ["lr", "rf", "lgbm"])
    if alg == "lr": p = {"alg": alg, "C": t.suggest_float("C", 1e-3, 1e2, log=True)}
    elif alg == "rf": p = {"alg": alg, "n": t.suggest_int("n_rf", 50, 200), "prof": t.suggest_int("prof", 3, 14)}
    else: p = {"alg": alg, "n": t.suggest_int("n_lgb", 50, 300), "lr": t.suggest_float("lr", 0.01, 0.3, log=True), "folhas": t.suggest_int("folhas", 4, 32)}
    return cross_val_score(constroi(p), Xtr, ytr, cv=3).mean()

def params(t):
    d = t.params; a = d["alg"]
    return {"alg": a, **({"C": d["C"]} if a == "lr" else {"n": d["n_rf"], "prof": d["prof"]} if a == "rf" else {"n": d["n_lgb"], "lr": d["lr"], "folhas": d["folhas"]})}

estudo = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0)); estudo.optimize(objetivo, n_trials=30)
ordem = sorted([t for t in estudo.trials if t.value is not None], key=lambda t: -t.value)
escolhidos, vistos = [], set()
for t in ordem:                                              # o melhor de cada algoritmo, para ter diversidade
    if t.params["alg"] not in vistos: vistos.add(t.params["alg"]); escolhidos.append(t)
modelos = [constroi(params(t)).fit(Xtr, ytr) for t in escolhidos]
print("Melhor da busca:", params(ordem[0]), "| CV = %.3f" % ordem[0].value)
print("Algoritmos no ensemble:", [params(t)["alg"] for t in escolhidos])
P = lambda ms, A: np.mean([m.predict_proba(A)[:, 1] for m in ms], axis=0)
print("Validação: melhor único = %.3f | ensemble = %.3f" % (modelos[0].score(Xv, yv), ((P(modelos, Xv) >= 0.5) == yv).mean()))

base_lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr); base_rf = RandomForestClassifier(100, random_state=0, n_jobs=-1).fit(Xtr, ytr)
pred_ens = (P(modelos, Xte) >= 0.5).astype(int)
cands = {"reg. logística (base)": base_lr.predict(Xte), "floresta padrão (base)": base_rf.predict(Xte), "melhor único": modelos[0].predict(Xte), "ensemble por algoritmo": pred_ens}
print("\nTESTE FINAL (n = %d, erro padrão ~ %.3f):" % (len(yte), np.sqrt(0.8 * 0.2 / len(yte))))
for nome, p in cands.items(): print(f"  {nome:<24s} acurácia = {(p == yte).mean():.3f}")
rng = np.random.default_rng(0)
def ic_pareado(pa, pb):                                     # diferença de acertos (a - b) e IC de 95% por bootstrap pareado
    d = (pa == yte).astype(int) - (pb == yte).astype(int); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
for nome in ("melhor único", "ensemble por algoritmo"):
    m, lo, hi = ic_pareado(cands[nome], cands["reg. logística (base)"])
    print(f"  {nome} - reg. logística: diferença = {m:+.4f} | IC 95% pareado = [{lo:+.4f}, {hi:+.4f}]")

def rel(nome, pred):
    s = [pred[gte == k].mean() for k in (0, 1)]; t = [pred[(gte == k) & (yrte == 1)].mean() for k in (0, 1)]
    print(f"  {nome:<22s} acc hist = {(pred == yte).mean():.3f} | acc verdade = {(pred == yrte).mean():.3f} | seleção {s[0]:.2f}/{s[1]:.2f} | TPR real {t[0]:.2f}/{t[1]:.2f}")
print("\nAUDITORIA POR GRUPO:"); rel("ensemble", pred_ens)
pi = permutation_importance(modelos[0], Xte, yte, n_repeats=10, random_state=0)
print("  Permutação do melhor único:", {n: round(float(v), 3) for n, v in zip(NOMES, pi.importances_mean)})
Xg_tr, Xg_te, g_tr, g_te = train_test_split(X, g, test_size=0.3, random_state=0, stratify=g)
pg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xg_tr, g_tr)
print("  AUC para prever o grupo pelos atributos: %.3f" % roc_auc_score(g_te, pg.predict_proba(Xg_te)[:, 1]))
pv, p = P(modelos, Xv), P(modelos, Xte)
sel0 = (pv[gv == 0] >= 0.5).mean(); t1 = np.quantile(pv[gv == 1], 1 - sel0)          # limiar escolhido na VALIDAÇÃO
rel("ensemble + limiar (valid.)", np.where(gte == 0, p >= 0.5, p >= t1).astype(int))

pasta = tempfile.mkdtemp(); joblib.dump(modelos[0], os.path.join(pasta, "modelo.joblib"))
ficha = {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__, "semente": 0, "hash_treino": hashlib.sha256(Xtr.tobytes()).hexdigest()[:16]}
um = Xte[:1]; modelos[0].predict(um); t0 = time.perf_counter()
for _ in range(50): modelos[0].predict(um)
lat = (time.perf_counter() - t0) / 50 * 1000; um_ens = time.perf_counter()
for _ in range(50): P(modelos, um)
lat_e = (time.perf_counter() - um_ens) / 50 * 1000
print("\nPRODUÇÃO: ficha =", json.dumps(ficha)); print(f"  Latência por previsão: melhor único = {lat:.1f} ms | ensemble = {lat_e:.1f} ms (orçamento: 20 ms)")
Xd, g_d, yh_d, yr_d = gera(3000, vies=0.5, p_min=0.25, seed=99, infl=0.4); Xd = Xd[:, [0, 1, 2, 4]]
print(f"  Deriva (inflação de 40%): PSI da renda = {psi(Xtr[:, 0], Xd[:, 0]):.3f} | acurácia contra a verdade: base = {(modelos[0].predict(Xte) == yrte).mean():.3f} -> {(modelos[0].predict(Xd) == yr_d).mean():.3f}")
