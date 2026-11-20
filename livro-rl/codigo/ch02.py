"""
Livro: Aprendizado por Reforço — Bandits Multi-Armados: Explorar ou Aproveitar?
Autor do livro: Araken de Medeiros Santos

Os blocos abaixo seguem a ordem do texto e devem ser executados em sequência (como células de um notebook).
Os trechos marcados como 'não executado' (por exemplo, downloads do Kaggle) aparecem comentados.
"""
import warnings; warnings.filterwarnings('ignore')

# ======================================================================
# Bloco 1
# ======================================================================
import numpy as np
import matplotlib.pyplot as plt

def desempate(M, rng):
    """argmax por linha, sorteando entre empates."""
    return (M + 1e-9 * rng.random(M.shape)).argmax(1)

def gulosa(Q, N, t, rng):
    return desempate(Q, rng)

def eps_gulosa(Q, N, t, rng, eps=0.1):
    a = desempate(Q, rng)
    sorteio = rng.random(len(a)) < eps
    a[sorteio] = rng.integers(Q.shape[1], size=sorteio.sum())
    return a

def ucb(Q, N, t, rng, c=2.0):
    bonus = np.where(N == 0, np.inf, c * np.sqrt(np.log(t + 1) / np.maximum(N, 1)))
    return desempate(Q + bonus, rng)

def executar(estrategia, q_verd, passos=1000, semente=1, **kw):
    rng = np.random.default_rng(semente)
    n, k = q_verd.shape; idx = np.arange(n)
    Q = np.zeros((n, k)); N = np.zeros((n, k))
    rec = np.zeros((n, passos)); otima = np.zeros((n, passos))
    melhor = q_verd.argmax(1)
    for t in range(passos):
        a = estrategia(Q, N, t, rng, **kw)
        r = rng.normal(q_verd[idx, a], 1.0)          # recompensa com ruído
        N[idx, a] += 1
        Q[idx, a] += (r - Q[idx, a]) / N[idx, a]     # atualização incremental
        rec[:, t] = r; otima[:, t] = (a == melhor)
    return rec.mean(0), otima.mean(0)

q = np.random.default_rng(0).normal(0, 1, (1000, 10))
for nome, f, kw in [("gulosa", gulosa, {}), ("ε = 0,01", eps_gulosa, {"eps": 0.01}),
                    ("ε = 0,1", eps_gulosa, {"eps": 0.1}), ("UCB c=2", ucb, {"c": 2.0})]:
    r, o = executar(f, q, **kw)
    print(f"{nome:10s} | recompensa média = {r.mean():.2f} | ação ótima nos últimos 100 passos = {100*o[-100:].mean():.0f}%")

# ======================================================================
# Bloco 2
# ======================================================================
def executar_bernoulli(estrategia, p, passos=2000, n_exec=300, semente=3):
    rng = np.random.default_rng(semente)
    p = np.asarray(p); k = len(p); idx = np.arange(n_exec)
    S = np.zeros((n_exec, k)); N = np.zeros((n_exec, k)); arrep = np.zeros((n_exec, passos))
    for t in range(passos):
        a = estrategia(S, N, t, rng)
        r = (rng.random(n_exec) < p[a]).astype(float)
        N[idx, a] += 1; S[idx, a] += r
        arrep[:, t] = p.max() - p[a]                    # perda esperada neste passo
    return np.cumsum(arrep.mean(0))

def b_eps(S, N, t, rng, eps=0.1):
    Q = np.where(N > 0, S / np.maximum(N, 1), 0.0)
    a = desempate(Q, rng); m = rng.random(len(a)) < eps
    a[m] = rng.integers(Q.shape[1], size=m.sum()); return a

def b_ucb(S, N, t, rng):
    Q = np.where(N > 0, S / np.maximum(N, 1), 0.0)
    b = np.where(N == 0, np.inf, np.sqrt(2 * np.log(t + 1) / np.maximum(N, 1)))
    return desempate(Q + b, rng)

def b_thompson(S, N, t, rng):
    return rng.beta(S + 1, N - S + 1).argmax(1)         # um sorteio da crença de cada braço

p = [0.45, 0.55, 0.60]
for nome, f in [("ε-gulosa", b_eps), ("UCB1", b_ucb), ("Thompson", b_thompson)]:
    curva = executar_bernoulli(f, p)
    print(f"{nome:9s} | arrependimento após 2000 passos = {curva[-1]:.1f}")

# ======================================================================
# Bloco 3
# ======================================================================
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

cogumelos = fetch_ucirepo(id=73)
X = cogumelos.data.features.fillna("ausente")
venenoso = (cogumelos.data.targets.iloc[:, 0] == "p").astype(int).to_numpy()
Xc = OneHotEncoder(handle_unknown="ignore").fit_transform(X).tocsr()
r_comer = np.where(venenoso == 1, -35.0, 5.0)        # recompensa de COMER cada cogumelo

def bandit_em_lotes(eps, n_lotes=20, tam=200, semente=0):
    rng = np.random.default_rng(semente)
    comidos, modelo, media_por_lote = [], None, []
    for _ in range(n_lotes):
        ids = rng.integers(0, Xc.shape[0], tam)
        p = np.zeros(tam) if modelo is None else modelo.predict_proba(Xc[ids])[:, 1]
        come = p < 0.125                                     # aproveitar
        explora = rng.random(tam) < eps
        come = np.where(explora, rng.random(tam) < 0.5, come)  # explorar
        media_por_lote.append(np.where(come, r_comer[ids], 0.0).mean())
        comidos.extend(ids[come])
        if len(set(venenoso[comidos])) == 2:                # precisa ter visto as duas classes
            modelo = LogisticRegression(max_iter=500).fit(Xc[comidos], venenoso[comidos])
    return np.array(media_por_lote)

for eps in [0.0, 0.05, 0.2]:
    R = np.array([bandit_em_lotes(eps, semente=s) for s in range(5)])
    print(f"ε = {eps:4.2f} | recompensa média por cogumelo, lotes 1, 2, 10 e 20:",
          R.mean(0)[[0, 1, 9, 19]].round(2), "| soma:", R.sum(1).mean().round(1))
print("Oráculo (come só os comestíveis):", np.maximum(r_comer, 0).mean().round(2))

# ======================================================================
# Bloco 4
# ======================================================================
banco = fetch_ucirepo(id=222)                         # Bank Marketing, UCI
df = banco.data.features.copy()
df["y"] = (banco.data.targets.iloc[:, 0] == "yes").astype(int)
taxas = df.groupby("month")["y"].mean().sort_values(ascending=False)
print(taxas.round(3))

p_meses = taxas.to_numpy()                            # 12 braços
for nome, f in [("ε-gulosa", b_eps), ("UCB1", b_ucb), ("Thompson", b_thompson)]:
    curva = executar_bernoulli(f, p_meses, passos=3000, n_exec=100)
    print(f"{nome:9s} | arrependimento final = {curva[-1]:.1f}")

