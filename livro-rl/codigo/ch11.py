"""
Livro: Aprendizado por Reforço — Aprendizado Baseado em Modelo e Planejamento
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

class Labirinto:
    L, C = 6, 9; inicio = (2, 0); meta = (0, 8)
    paredes = {(1, 2), (2, 2), (3, 2), (4, 5), (0, 7), (1, 7), (2, 7)}
    acoes = [(-1, 0), (1, 0), (0, -1), (0, 1)]                 # cima, baixo, esquerda, direita
    def passo(self, s, a):
        i, j = s[0] + self.acoes[a][0], s[1] + self.acoes[a][1]
        if not (0 <= i < self.L and 0 <= j < self.C) or (i, j) in self.paredes: i, j = s
        s2 = (i, j); return s2, (1.0 if s2 == self.meta else 0.0), s2 == self.meta

def dyna_q(n_plan, n_ep=50, alpha=0.1, gamma=0.95, eps=0.1, semente=0):
    rng = np.random.default_rng(semente); env = Labirinto(); Q = {}; modelo = {}; passos = []
    q = lambda s: Q.setdefault(s, np.zeros(4))
    for ep in range(n_ep):
        s = env.inicio; n = 0; fim = False
        while not fim:
            a = int(rng.integers(4)) if rng.random() < eps else int(np.argmax(q(s) + 1e-9 * rng.random(4)))
            s2, r, fim = env.passo(s, a); n += 1
            q(s)[a] += alpha * (r + gamma * q(s2).max() * (not fim) - q(s)[a])     # 1) aprendizado direto
            modelo[(s, a)] = (r, s2, fim)                                           # 2) aprende o modelo (determinístico)
            chaves = list(modelo.keys())
            for _ in range(n_plan):                                                 # 3) planejamento: n passos imaginados
                (ps, pa) = chaves[rng.integers(len(chaves))]; pr, ps2, pf = modelo[(ps, pa)]
                q(ps)[pa] += alpha * (pr + gamma * q(ps2).max() * (not pf) - q(ps)[pa])
            s = s2
        passos.append(n)
    return np.array(passos)

for n in [0, 5, 50]:
    P = np.mean([dyna_q(n, semente=k) for k in range(5)], axis=0)
    print(f"n = {n:2d} passos de planejamento | passos até a meta nos episódios 1, 2, 5, 10, 50: {P[[0, 1, 4, 9, 49]].round(0)}")

# ======================================================================
# Bloco 2
# ======================================================================
from sklearn.ensemble import ExtraTreesRegressor

env = gym.make("CartPole-v1"); rng = np.random.default_rng(0)

def coletar(n, semente=0):
    r_ = np.random.default_rng(semente); O, A, O2 = [], [], []; o, _ = env.reset(seed=semente)
    for _ in range(n):
        a = int(r_.integers(2)); o2, r, term, trunc, _ = env.step(a); O.append(o); A.append(a); O2.append(o2); o = o2
        if term or trunc: o, _ = env.reset()
    return np.array(O), np.array(A), np.array(O2)

O, A, O2 = coletar(5000, 0)
modelo_cp = ExtraTreesRegressor(n_estimators=50, min_samples_leaf=2, random_state=0, n_jobs=1).fit(np.column_stack([O, A]), O2 - O)

# erro de previsão de vários passos, em trajetórias novas (ações aleatórias)
H = 15; trajs = []; o, _ = env.reset(seed=123); traj, acs = [o], []
for _ in range(1500):
    a = int(rng.integers(2)); o2, r, term, trunc, _ = env.step(a); acs.append(a); traj.append(o2)
    if term or trunc: trajs.append((np.array(traj), np.array(acs))); o, _ = env.reset(); traj, acs = [o], []
erros = [[] for _ in range(H)]
for tj, ac in trajs:
    for t0 in range(0, len(ac) - H, 3):
        s = tj[t0].copy()
        for h in range(H):
            s = s + modelo_cp.predict(np.append(s, ac[t0 + h]).reshape(1, -1))[0]; erros[h].append(np.abs(s - tj[t0 + h + 1]))
E = np.array([np.mean(e, axis=0) for e in erros])
print("erro absoluto médio do ângulo do poste (rad), por horizonte:", E[[0, 4, 9, 14], 2].round(4), "(passos 1, 5, 10, 15)")

# ======================================================================
# Bloco 3
# ======================================================================
def terminou(s):
    return (np.abs(s[:, 0]) > 2.4) | (np.abs(s[:, 2]) > 0.2095)

def planejar(modelo, s0, rng_l, H=15, N=200):
    seqs = rng_l.integers(0, 2, (N, H)); s = np.tile(s0, (N, 1)); vivo = np.ones(N, bool); ret = np.zeros(N)
    for h in range(H):
        s = s + modelo.predict(np.column_stack([s, seqs[:, h]]))     # todas as sequências, no modelo
        ret += vivo; vivo &= ~terminou(s)                              # +1 por passo em que ainda "vive"
    return int(seqs[np.argmax(ret), 0])                                # só a primeira ação da melhor sequência

rng_p = np.random.default_rng(1); R = []
for ep in range(3):
    o, _ = env.reset(seed=700 + ep); total, fim = 0, False
    while not fim:
        o, r, term, trunc, _ = env.step(planejar(modelo_cp, o, rng_p)); total += r; fim = term or trunc
    R.append(total)
print(f"MPC com modelo aprendido de 5.000 transições aleatórias: retorno médio = {np.mean(R):.1f} (agente aleatório ≈ 22)")

# ======================================================================
# Bloco 4
# ======================================================================
import math, random

LINHAS = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
def vencedor(t):                                   # 1 ou 2 = jogador vencedor; 3 = empate; 0 = em andamento
    for a, b, c in LINHAS:
        if t[a] != 0 and t[a] == t[b] == t[c]: return t[a]
    return 0 if 0 in t else 3
def jogadas(t): return [i for i in range(9) if t[i] == 0]

class No:
    def __init__(self, t, jogador, pai=None, mov=None):
        self.t, self.j, self.pai, self.mov = t, jogador, pai, mov
        self.filhos = []; self.nao_exp = jogadas(t) if vencedor(t) == 0 else []; self.n = 0; self.w = 0.0

def uct(no, c=1.4):
    return max(no.filhos, key=lambda f: f.w / f.n + c * math.sqrt(math.log(no.n) / f.n))

def mcts(t, jogador, n_sim=500):
    raiz = No(t, jogador)
    for _ in range(n_sim):
        no = raiz
        while not no.nao_exp and no.filhos: no = uct(no)                              # 1) seleção
        if no.nao_exp:                                                                # 2) expansão
            m = no.nao_exp.pop(random.randrange(len(no.nao_exp))); t2 = list(no.t); t2[m] = no.j
            f = No(tuple(t2), 3 - no.j, no, m); no.filhos.append(f); no = f
        tt = list(no.t); j = no.j; v = vencedor(tuple(tt))                            # 3) simulação
        while v == 0:
            m = random.choice(jogadas(tt)); tt[m] = j; j = 3 - j; v = vencedor(tuple(tt))
        while no is not None:                                                         # 4) retropropagação
            quem = 3 - no.j                                                           # quem jogou para chegar a este nó
            no.n += 1; no.w += 0.5 if v == 3 else (1.0 if v == quem else 0.0); no = no.pai
    return max(raiz.filhos, key=lambda f: f.n).mov

def partida(mcts_primeiro, n_sim):
    t = [0] * 9; j = 1
    while vencedor(tuple(t)) == 0:
        m = mcts(tuple(t), j, n_sim) if (j == 1) == mcts_primeiro else random.choice(jogadas(t))
        t[m] = j; j = 3 - j
    return vencedor(tuple(t))

random.seed(0)
for n_sim in [5, 50, 500]:
    v = e = d = 0
    for k in range(40):
        r = partida(k % 2 == 0, n_sim); meu = 1 if k % 2 == 0 else 2
        if r == 3: e += 1
        elif r == meu: v += 1
        else: d += 1
    print(f"{n_sim:3d} simulações por jogada | em 40 partidas: {v} vitórias, {e} empates, {d} derrotas")

# ======================================================================
# Bloco 5
# ======================================================================
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

painel = pd.read_csv("painel_clientes.csv").sort_values(["cliente", "t"])
for c in ["rec", "freq", "mon"]:
    painel[c + "_prox"] = painel.groupby("cliente")[c].shift(-1)
d = painel.dropna(subset=["rec_prox", "gasto_prox"]).copy()
X = d[["rec", "freq", "mon"]].to_numpy(); Y = d[["rec_prox", "freq_prox", "mon_prox", "gasto_prox"]].to_numpy()
grupos = d["cliente"].to_numpy(); tr, te = next(GroupKFold(n_splits=5).split(X, groups=grupos))
mundo = ExtraTreesRegressor(n_estimators=100, min_samples_leaf=5, random_state=0, n_jobs=1).fit(X[tr], Y[tr])
print("MAE de 1 passo no teste (rec, freq, mon, gasto):", mean_absolute_error(Y[te], mundo.predict(X[te]), multioutput="raw_values").round(2))

residuos = Y[tr] - mundo.predict(X[tr]); rng_c = np.random.default_rng(0)
def simular_cliente(s0, H=6):
    s = np.array([s0], float); gasto = 0.0; acumulado = []
    for _ in range(H):
        p = mundo.predict(s)[0] + residuos[rng_c.integers(len(residuos))]       # previsão + ruído
        gasto += max(p[3], 0.0); acumulado.append(gasto); s = np.array([[max(p[0], 0), max(p[1], 0), max(p[2], 0)]])
    return np.array(acumulado)

reais, previstos = [], []
for cli, g in d.iloc[te].groupby("cliente"):
    g = g.sort_values("t")
    if len(g) >= 6:
        reais.append(np.cumsum(g["gasto_prox"].to_numpy()[:6]))
        previstos.append(np.mean([simular_cliente(g[["rec", "freq", "mon"]].to_numpy()[0]) for _ in range(5)], axis=0))
reais, previstos = np.array(reais), np.array(previstos)
print("clientes comparados:", len(reais), "| erro absoluto médio do gasto acumulado, mês 1 a 6:", np.abs(reais - previstos).mean(0).round(1))

