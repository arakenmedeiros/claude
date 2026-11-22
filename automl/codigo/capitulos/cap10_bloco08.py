# Capítulo 10 — Ferramentas de AutoML na prática
# Bloco 8. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
import warnings; warnings.filterwarnings("ignore")
import json, os, resource, time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np

METODOS = ["Floresta padrão", "Boosting padrão", "Aleatória (10 s)", "TPE (10 s)", "FLAML (10 s)"]
BASES = ["Wine", "Cancer", "Digits", "Ruidosa"]

def carrega(base):
    from sklearn.datasets import load_wine, load_breast_cancer, load_digits, make_classification
    if base == "Ruidosa": return make_classification(1500, 30, n_informative=8, n_redundant=4, flip_y=0.10, random_state=0)
    return {"Wine": load_wine, "Cancer": load_breast_cancer, "Digits": load_digits}[base](return_X_y=True)

def executa(metodo, base, semente):
    """Roda UM método em UM processo novo, com 1 núcleo, e devolve a acurácia no teste e os recursos consumidos."""
    from threadpoolctl import threadpool_limits
    import optuna
    from flaml import AutoML
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X, y = carrega(base)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=semente)
    t0, c0 = time.perf_counter(), time.process_time(); n_conf = 1
    with threadpool_limits(limits=1):                                          # 1 núcleo para todos
        if metodo == "Floresta padrão":
            m = RandomForestClassifier(200, random_state=semente, n_jobs=1).fit(Xtr, ytr)
        elif metodo == "Boosting padrão":
            m = HistGradientBoostingClassifier(random_state=semente).fit(Xtr, ytr)
        elif metodo == "FLAML (10 s)":
            m = AutoML()
            m.fit(Xtr, ytr, task="classification", time_budget=10, metric="accuracy", eval_method="cv",
                  n_splits=3, seed=semente, n_jobs=1, verbose=0)
            n_conf = sum(s.total_iter for s in m._search_states.values())        # atributo interno do FLAML
        else:                                                                  # Optuna: mesmo espaço (boosting), mesmo tempo
            def cria(t):
                return HistGradientBoostingClassifier(
                    learning_rate=t.suggest_float("learning_rate", 0.01, 0.5, log=True), max_iter=t.suggest_int("max_iter", 50, 300),
                    max_leaf_nodes=t.suggest_int("max_leaf_nodes", 4, 40), min_samples_leaf=t.suggest_int("min_samples_leaf", 5, 60),
                    l2_regularization=t.suggest_float("l2", 1e-4, 10, log=True), random_state=semente)
            amostrador = optuna.samplers.RandomSampler(seed=semente) if metodo.startswith("Aleat") else optuna.samplers.TPESampler(seed=semente)
            estudo = optuna.create_study(direction="maximize", sampler=amostrador)
            estudo.optimize(lambda t: cross_val_score(cria(t), Xtr, ytr, cv=3).mean(), timeout=10)
            m = cria(estudo.best_trial).fit(Xtr, ytr); n_conf = len(estudo.trials)
    return {"acc": float((m.predict(Xte) == yte).mean()), "tempo": time.perf_counter() - t0, "cpu": time.process_time() - c0,
            "configs": int(n_conf), "mem_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "escolhido": getattr(m, "best_estimator", "")}

if __name__ == "__main__":                                                    # necessário para processos novos (spawn)
    ctx = mp.get_context("spawn"); R = {}
    for base in BASES:
        for metodo in METODOS:
            for s in (0, 1, 2):
                with ProcessPoolExecutor(1, mp_context=ctx, max_tasks_per_child=1) as ex:      # um processo novo por execução
                    R[f"{base}|{metodo}|{s}"] = ex.submit(executa, metodo, base, s).result()
    res = {b: {m: [R[f"{b}|{m}|{s}"]["acc"] for s in (0, 1, 2)] for m in METODOS} for b in BASES}
    print("Acurácia média no teste (3 divisões):")
    print(f"{'base':<8s}" + "".join(f"{m.split(' (')[0][:9]:>11s}" for m in METODOS))
    for b in BASES: print(f"{b:<8s}" + "".join(f"{np.mean(res[b][m]):11.3f}" for m in METODOS))
    print("\nRecursos médios por execução (1 núcleo; um processo novo por execução):")
    print(f"{'método':<18s} {'tempo (s)':>9s} {'CPU (s)':>8s} {'configs':>8s} {'pico mem. (MB)':>15s}")
    rec = {}
    for m in METODOS:
        L = [v for k, v in R.items() if k.split("|")[1] == m]
        rec[m] = {c: float(np.mean([v[c] for v in L])) for c in ("tempo", "cpu", "configs", "mem_mb")}
        print(f"{m:<18s} {rec[m]['tempo']:9.1f} {rec[m]['cpu']:8.1f} {rec[m]['configs']:8.1f} {rec[m]['mem_mb']:15.0f}")
    os.makedirs("resultados", exist_ok=True)
    json.dump({"res": res, "metodos": METODOS, "recursos": rec,
               "escolhido": {b: [R[f"{b}|FLAML (10 s)|{s}"]["escolhido"] for s in (0, 1, 2)] for b in BASES}}, open("resultados/f10_cache.json", "w"))
