"""
Livro: Aprendizado por Reforço — Programação Dinâmica
Autor do livro: Araken de Medeiros Santos

Os blocos abaixo seguem a ordem do texto e devem ser executados em sequência (como células de um notebook).
Os trechos marcados como 'não executado' (por exemplo, downloads do Kaggle) aparecem comentados.
"""
import warnings; warnings.filterwarnings('ignore')

# ======================================================================
# Bloco 1
# ======================================================================
import numpy as np

def criar_grade(n=4):
    """Gridworld n×n: terminais nos cantos opostos; -1 por passo; ações 0=↑ 1=↓ 2=← 3=→."""
    nS = n * n; P = np.zeros((nS, 4, nS)); R = -np.ones((nS, 4))
    terminais = [0, nS - 1]
    for s in range(nS):
        if s in terminais:
            P[s, :, s] = 1; R[s, :] = 0; continue
        i, j = divmod(s, n)
        for a, (di, dj) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
            i2, j2 = min(max(i + di, 0), n - 1), min(max(j + dj, 0), n - 1)
            P[s, a, i2 * n + j2] = 1
    return P, R

def avaliar_politica(P, R, pol, gamma=1.0, theta=1e-8):
    V = np.zeros(P.shape[0])
    for k in range(1, 100000):
        V_novo = np.einsum("sa,sa->s", pol, R + gamma * P @ V)   # Bellman de expectativa
        delta = np.abs(V_novo - V).max(); V = V_novo
        if delta < theta: break
    return V, k

def iteracao_de_politica(P, R, gamma=1.0):
    nS, nA = R.shape; pol = np.full((nS, nA), 1 / nA); total = 0
    for it in range(1, 1000):
        V, k = avaliar_politica(P, R, pol, gamma); total += k
        Q = R + gamma * P @ V
        nova = np.zeros_like(pol); nova[np.arange(nS), Q.argmax(1)] = 1   # gulosa
        if np.array_equal(nova, pol): return V, pol, it, total
        pol = nova

def iteracao_de_valor(P, R, gamma=1.0, theta=1e-8):
    V = np.zeros(R.shape[0])
    for k in range(1, 100000):
        V_novo = (R + gamma * P @ V).max(axis=1)                          # Bellman de otimalidade
        delta = np.abs(V_novo - V).max(); V = V_novo
        if delta < theta: break
    Q = R + gamma * P @ V
    pol = np.zeros_like(Q); pol[np.arange(len(V)), Q.argmax(1)] = 1
    return V, pol, k

P, R = criar_grade(4)
V_rand, k = avaliar_politica(P, R, np.full((16, 4), 0.25))
print(f"Política aleatória: convergiu em {k} varreduras")
print(V_rand.reshape(4, 4).round(1))

V_pi, pol_pi, n_ext, n_total = iteracao_de_politica(P, R)
V_vi, pol_vi, n_var = iteracao_de_valor(P, R)
print(f"\nIteração de política: {n_ext} iterações externas ({n_total} varreduras no total)")
print(f"Iteração de valor: {n_var} varreduras")
print("Valores ótimos (negativo do nº mínimo de passos):"); print(V_vi.reshape(4, 4).round(0))
print("Os dois métodos concordam?", np.allclose(V_pi, V_vi))

# ======================================================================
# Bloco 2
# ======================================================================
import gymnasium as gym
from sklearn.utils import resample

env = gym.make("FrozenLake-v1", is_slippery=True)
Pf = env.unwrapped.P; nS, nA = env.observation_space.n, env.action_space.n
Pm = np.zeros((nS, nA, nS)); Rm = np.zeros((nS, nA))
for s in range(nS):
    for a in range(nA):
        for prob, s2, r, fim in Pf[s][a]:
            Pm[s, a, s2] += prob; Rm[s, a] += prob * r

V_otimo, pol_otima, n_var = iteracao_de_valor(Pm, Rm, gamma=0.99, theta=1e-10)
print(f"Iteração de valor convergiu em {n_var} varreduras")
print("Valor ótimo de cada casa:"); print(V_otimo.reshape(4, 4).round(2))
print("Ações ótimas (0=← 1=↓ 2=→ 3=↑):"); print(pol_otima.argmax(1).reshape(4, 4))

def jogar(politica, n=2000, semente=0):
    rng = np.random.default_rng(semente); sucessos = []
    for _ in range(n):
        s, _ = env.reset(seed=int(rng.integers(1_000_000))); fim = False
        while not fim:
            s, r, term, trunc, _ = env.step(int(politica[s]))
            fim = term or trunc
        sucessos.append(r)                      # recompensa 1 só se chegou à meta
    return np.array(sucessos)

suc = jogar(pol_otima.argmax(1))
boot = [resample(suc, random_state=b).mean() for b in range(500)]
print(f"Taxa de sucesso da política ótima: {suc.mean():.3f} (IC 95%: {np.percentile(boot, 2.5):.3f} a {np.percentile(boot, 97.5):.3f})")
aleatoria = jogar(np.random.default_rng(1).integers(0, 4, nS))
print(f"Política sorteada (referência): {aleatoria.mean():.3f}")

# ======================================================================
# Bloco 3
# ======================================================================
from sklearn.preprocessing import KBinsDiscretizer

env = gym.make("CartPole-v1"); rng = np.random.default_rng(0)

# 1) Coleta aleatória de transições
O, A, O2, F = [], [], [], []
o, _ = env.reset(seed=0)
for _ in range(150_000):
    a = int(rng.integers(2)); o2, r, term, trunc, _ = env.step(a)
    O.append(o); A.append(a); O2.append(o2); F.append(term)
    o = o2
    if term or trunc: o, _ = env.reset()
O, A, O2, F = map(np.array, (O, A, O2, F))

# 2) Discretização em faixas (6 × 6 × 12 × 12 = 5184 estados)
lim = np.array([2.4, 3.0, 0.21, 3.0])
disc = KBinsDiscretizer(n_bins=[6, 6, 12, 12], encode="ordinal", strategy="uniform")
disc.fit(np.vstack([-lim, lim]))
def para_estado(X):
    idx = disc.transform(np.clip(X, -lim + 1e-6, lim - 1e-6)).astype(int)
    return np.ravel_multi_index(idx.T, disc.n_bins_.astype(int))

S, S2 = para_estado(O), para_estado(O2)
nS_c = int(np.prod(disc.n_bins_))
cont = np.zeros((nS_c, 2, nS_c)); np.add.at(cont, (S, A, S2), 1)     # contagem de transições
quedas = np.zeros((nS_c, 2)); np.add.at(quedas, (S, A), F.astype(float))
N_sa = cont.sum(axis=2)
vis = np.where(N_sa.sum(1) > 0)[0]                                     # estados visitados
print(f"Estados visitados: {len(vis)} de {nS_c}")
P_hat = cont[vis] / np.maximum(N_sa[vis][:, :, None], 1)               # probabilidades estimadas
p_queda = quedas[vis] / np.maximum(N_sa[vis], 1)                       # prob. de o poste cair

# 3) Iteração de valor no modelo estimado (recompensa +1 por passo em que não cai)
gamma = 0.99; V = np.zeros(nS_c)
for k in range(5000):
    Q_vis = 1.0 + gamma * (1 - p_queda) * (P_hat @ V)
    Q_vis[N_sa[vis] == 0] = 0.0                                        # par nunca visto: pessimista
    V_novo = V.copy(); V_novo[vis] = Q_vis.max(axis=1)
    if np.abs(V_novo - V).max() < 1e-4: V = V_novo; break
    V = V_novo
acao_de = {s: int(Q_vis[i].argmax()) for i, s in enumerate(vis)}
print(f"Iteração de valor convergiu em {k} varreduras")

# 4) Teste no ambiente real
def politica_dp(o):
    return acao_de.get(int(para_estado(o[None])[0]), int(rng.integers(2)))
ret = []
for ep in range(50):
    o, _ = env.reset(seed=1000 + ep); total, fim = 0, False
    while not fim:
        o, r, term, trunc, _ = env.step(politica_dp(o)); total += r; fim = term or trunc
    ret.append(total)
print(f"Retorno médio da política planejada: {np.mean(ret):.1f}  (mín {min(ret):.0f}, máx {max(ret):.0f})")

# ======================================================================
# Bloco 4
# ======================================================================
from sklearn.model_selection import ParameterGrid

P_cli = np.load("P_clientes.npy"); r_cli = np.load("r_clientes.npy"); K = len(r_cli)

def mdp_cupom(c, delta):
    P2 = np.zeros((K, 2, K)); R2 = np.zeros((K, 2))
    P2[:, 0, :] = P_cli; R2[:, 0] = r_cli                 # ação 0: sem cupom
    for s in range(K):
        p = P_cli[s].copy(); topo = K - 1
        p = p * (1 - delta); p[topo] += delta             # empurra delta de massa para o melhor estado
        P2[s, 1, :] = p; R2[s, 1] = r_cli[s] - c          # ação 1: com cupom (custa c)
    return P2, R2

linhas = []
for params in ParameterGrid({"c": [1, 3, 5, 10], "delta": [0.05, 0.10, 0.20]}):
    P2, R2 = mdp_cupom(**params)
    V_c, pol_c, _ = iteracao_de_valor(P2, R2, gamma=0.95, theta=1e-9)
    linhas.append({**params, "cupom_nos_estados": [int(s) for s in range(K) if pol_c[s, 1] == 1]})
import pandas as pd
print(pd.DataFrame(linhas))

