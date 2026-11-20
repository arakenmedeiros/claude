"""
Livro: Aprendizado por Reforço — Aprendizado Offline, por Imitação e Avaliação de Políticas
Autor do livro: Araken de Medeiros Santos

Os blocos abaixo seguem a ordem do texto e devem ser executados em sequência (como células de um notebook).
Os trechos marcados como 'não executado' (por exemplo, downloads do Kaggle) aparecem comentados.
"""
import warnings; warnings.filterwarnings('ignore')

# ======================================================================
# Bloco 1
# ======================================================================
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

envs = make_vec_env("CartPole-v1", n_envs=4, seed=0)
especialista = PPO("MlpPolicy", envs, n_steps=256, batch_size=64, seed=0, verbose=0).learn(total_timesteps=40_000)
env = gym.make("CartPole-v1")
print("Especialista (PPO):", evaluate_policy(especialista, env, n_eval_episodes=10)[0])

def demonstracoes(n_passos, politica, semente=0, eps=0.0):
    rng = np.random.default_rng(semente); S, A = [], []; o, _ = env.reset(seed=semente)
    for _ in range(n_passos):
        a = int(politica(o)) if rng.random() >= eps else int(rng.integers(2))
        S.append(o); A.append(a); o, r, term, trunc, _ = env.step(a)
        if term or trunc: o, _ = env.reset()
    return np.array(S), np.array(A)

pol_esp = lambda o: especialista.predict(o, deterministic=True)[0]

# ======================================================================
# Bloco 2
# ======================================================================
from sklearn.tree import DecisionTreeClassifier

def avaliar(politica, n=20, semente=900):
    ret = []
    for i in range(n):
        o, _ = env.reset(seed=semente + i); total, fim = 0, False
        while not fim:
            o, r, term, trunc, _ = env.step(int(politica(o))); total += r; fim = term or trunc
        ret.append(total)
    return np.mean(ret)

for n in [200, 1000, 5000]:
    S, A = demonstracoes(n, pol_esp)
    arv = DecisionTreeClassifier(max_depth=8, random_state=0).fit(S, A)
    print(f"BC com {n:5d} demonstrações | acerto (treino) = {arv.score(S, A):.3f} | retorno da política clonada = {avaliar(lambda o: arv.predict(o.reshape(1, -1))[0]):6.1f}")

# ======================================================================
# Bloco 3
# ======================================================================
def dagger(n_rodadas=5, passos_por_rodada=500):
    S, A = demonstracoes(500, pol_esp, semente=1)
    for rodada in range(n_rodadas):
        arv = DecisionTreeClassifier(max_depth=8, random_state=0).fit(S, A)
        aprendiz = lambda o, arv=arv: arv.predict(o.reshape(1, -1))[0]
        # executa o APRENDIZ e pede ao especialista os rótulos dos estados visitados
        Sn, _ = demonstracoes(passos_por_rodada, aprendiz, semente=10 + rodada)
        An = np.array([pol_esp(s) for s in Sn])
        S, A = np.vstack([S, Sn]), np.concatenate([A, An])
        print(f"rodada {rodada+1} | dados = {len(S):5d} | retorno do aprendiz = {avaliar(aprendiz):6.1f}")
dagger()

# ======================================================================
# Bloco 4
# ======================================================================
from sklearn.ensemble import ExtraTreesRegressor

def conjunto_misto(n, semente=0, eps=0.4):
    rng = np.random.default_rng(semente); O, A, R, O2, T = [], [], [], [], []; o, _ = env.reset(seed=semente)
    for _ in range(n):
        a = int(rng.integers(2)) if rng.random() < eps else int(pol_esp(o))
        o2, r, term, trunc, _ = env.step(a); O.append(o); A.append(a); R.append(r); O2.append(o2); T.append(term); o = o2
        if term or trunc: o, _ = env.reset()
    return map(np.array, (O, A, R, O2, T))

O, A, R, O2, T = conjunto_misto(15_000)
print(f"retorno médio do comportamento nos dados (aprox.): {avaliar(lambda o: pol_esp(o) if np.random.rand() > 0.4 else np.random.randint(2)):.1f}")

arv_bc = DecisionTreeClassifier(max_depth=8, random_state=0).fit(O, A)
print("BC sobre o conjunto misto:", round(avaliar(lambda o: arv_bc.predict(o.reshape(1, -1))[0]), 1))

X = np.column_stack([O, A]); modelo = None
for it in range(15):
    if modelo is None: y = R.astype(float)
    else:
        n = len(O2); Xb = np.vstack([np.column_stack([O2, np.zeros(n)]), np.column_stack([O2, np.ones(n)])]); q = modelo.predict(Xb)
        y = R + 0.99 * np.maximum(q[:n], q[n:]) * (~T)
    modelo = ExtraTreesRegressor(n_estimators=30, min_samples_leaf=5, random_state=0).fit(X, y)
print("FQI offline:", round(avaliar(lambda o: int(np.argmax(modelo.predict(np.array([[*o, 0], [*o, 1]]))))), 1))

# ======================================================================
# Bloco 5
# ======================================================================
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier

banco = fetch_ucirepo(id=222)
Xb = banco.data.features.copy(); yb = (banco.data.targets.iloc[:, 0] == "yes").astype(int).to_numpy()
m = Xb["contact"].isin(["cellular", "telephone"]).to_numpy()
Xb = Xb[m].reset_index(drop=True); yb = yb[m]; ab = (Xb["contact"] == "cellular").astype(int).to_numpy()
Xb = Xb.drop(columns=["contact", "duration"], errors="ignore")              # duration vaza informação do resultado
cat = Xb.select_dtypes(exclude="number").columns.tolist(); num = Xb.select_dtypes(include="number").columns.tolist(); Xb[cat] = Xb[cat].fillna("desconhecido")
prep = ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat), ("n", StandardScaler(), num)])
Z = prep.fit_transform(Xb).astype(float); n = len(Z)
print(f"n = {n} | fração contatada por celular = {ab.mean():.2f} | taxa de sucesso observada = {yb.mean():.3f}")

cv = StratifiedKFold(5, shuffle=True, random_state=0)
e = cross_val_predict(LogisticRegression(max_iter=1000), Z, ab, cv=cv, method="predict_proba")[:, 1]
e = np.clip(e, 0.05, 0.95)                                                  # propensão estimada, recortada para evitar pesos enormes
mu = np.column_stack([LogisticRegression(max_iter=1000).fit(Z[ab == k], yb[ab == k]).predict_proba(Z)[:, 1] for k in (0, 1)])

def estimadores(pi, a, y, e, mu):
    idx = np.arange(len(pi)); p_a = np.where(a == 1, e, 1 - e); w = (a == pi) / p_a
    dm = mu[idx, pi].mean(); ips = (w * y).mean(); snips = (w * y).sum() / w.sum()
    dr = (mu[idx, pi] + w * (y - mu[idx, a])).mean()
    return dict(DM=dm, IPS=ips, SNIPS=snips, DR=dr)

politicas = {"sempre celular": np.ones(n, int), "sempre telefone fixo": np.zeros(n, int), "melhor segundo o modelo": (mu[:, 1] > mu[:, 0]).astype(int)}
for nome, pi in politicas.items():
    est = estimadores(pi, ab, yb, e, mu); print(f"{nome:24s}", {k: round(float(v), 4) for k, v in est.items()})

# ======================================================================
# Bloco 6
# ======================================================================
mu_v = np.column_stack([HistGradientBoostingClassifier(max_depth=3, max_iter=60, random_state=0).fit(Z[ab == k], yb[ab == k]).predict_proba(Z)[:, 1] for k in (0, 1)])
pi_alvo = (mu_v[:, 1] > mu_v[:, 0]).astype(int); verdade = mu_v[np.arange(n), pi_alvo].mean()
rng = np.random.default_rng(0); erros = {k: [] for k in ["DM", "IPS", "SNIPS", "DR"]}
for rep in range(30):
    a_s = (rng.random(n) < e).astype(int); y_s = (rng.random(n) < mu_v[np.arange(n), a_s]).astype(int)
    mu_hat = np.column_stack([LogisticRegression(max_iter=500).fit(Z[a_s == k], y_s[a_s == k]).predict_proba(Z)[:, 1] for k in (0, 1)])
    est = estimadores(pi_alvo, a_s, y_s, e, mu_hat)
    for k in erros: erros[k].append(est[k] - verdade)
print(f"valor verdadeiro da política-alvo: {verdade:.4f}")
for k, v in erros.items():
    print(f"{k:5s} | viés = {np.mean(v):+.4f} | desvio-padrão = {np.std(v):.4f} | RMSE = {np.sqrt(np.mean(np.square(v))):.4f}")

# ======================================================================
# Bloco 7
# ======================================================================
import numpy as np

# MDP simples: estado = posição numa reta (0..4); ação 1 = "avançar" (sobe com prob. 0,8), ação 0 = "recuar".
# Recompensa = +1 ao estar no estado 4, 0 caso contrário. O comportamento sorteia ação 1 com prob. 0,5; a política-alvo, com 0,8.
def simular_seq(p_acao1, H, n, rng, politica_log=None):
    S = np.zeros((n, H + 1), int); A = np.zeros((n, H), int); Rw = np.zeros((n, H))
    for i in range(n):
        s = 2
        for t in range(H):
            a = int(rng.random() < p_acao1); A[i, t] = a; S[i, t] = s
            if a == 1: s = min(4, s + 1) if rng.random() < 0.8 else max(0, s - 1)
            else: s = max(0, s - 1) if rng.random() < 0.8 else min(4, s + 1)
            Rw[i, t] = float(s == 4)
    return A, Rw

def razoes_seq(A, p_b, p_e):
    pb = np.where(A == 1, p_b, 1 - p_b); pe = np.where(A == 1, p_e, 1 - p_e); return pe / pb

def estimadores_seq(A, Rw, p_b, p_e, gamma=1.0):
    rho = razoes_seq(A, p_b, p_e); H = A.shape[1]; desc = gamma ** np.arange(H)
    rho_cum = np.cumprod(rho, axis=1)                       # razão acumulada até o passo t
    G = (Rw * desc).sum(axis=1)
    is_traj = rho_cum[:, -1] * G                            # IS: uma razão para a trajetória inteira
    pdis = (rho_cum * Rw * desc).sum(axis=1)                # PDIS: cada recompensa usa só a razão até o seu passo
    w = rho_cum[:, -1]; wis = (w * G).sum() / w.sum()       # IS auto-normalizado (ponderado)
    ess = w.sum() ** 2 / (w ** 2).sum()                     # tamanho efetivo da amostra
    return is_traj.mean(), pdis.mean(), wis, ess

p_b, p_e, rng = 0.5, 0.8, np.random.default_rng(0)
print(f"{'H':>3} | {'verdade':>8} | {'IS':>14} | {'PDIS':>14} | {'WIS':>14} | {'ESS médio':>9}")
for H in (3, 5, 10, 20, 40):
    A, Rw = simular_seq(p_e, H, 20000, rng); verdade = Rw.sum(axis=1).mean()
    rep = []
    for _ in range(200):
        A, Rw = simular_seq(p_b, H, 300, rng); rep.append(estimadores_seq(A, Rw, p_b, p_e))
    rep = np.array(rep); rmse = np.sqrt(((rep[:, :3] - verdade) ** 2).mean(axis=0))
    print(f"{H:3d} | {verdade:8.2f} | RMSE {rmse[0]:7.2f} | RMSE {rmse[1]:7.2f} | RMSE {rmse[2]:7.2f} | {rep[:, 3].mean():9.1f}")

# ======================================================================
# Bloco 8
# ======================================================================
from sklearn.utils import resample

pi = politicas["sempre celular"]
def dr_boot(semente):
    idx = resample(np.arange(n), random_state=semente)
    return estimadores(pi[idx], ab[idx], yb[idx], e[idx], mu[idx])["DR"]
valores = [dr_boot(b) for b in range(200)]
print(f"DR (sempre celular) = {estimadores(pi, ab, yb, e, mu)['DR']:.4f} | IC 95%: {np.percentile(valores, 2.5):.4f} a {np.percentile(valores, 97.5):.4f}")
print(f"clientes com propensão extrema (<0,1 ou >0,9): {np.mean((e < 0.1) | (e > 0.9)):.1%}")

