import math, random, numpy as np, gymnasium as gym
from sklearn.ensemble import ExtraTreesRegressor
# ---------- Dyna maze
class Labirinto:
    L, C = 6, 9; inicio = (2, 0); meta = (0, 8); paredes = {(1, 2), (2, 2), (3, 2), (4, 5), (0, 7), (1, 7), (2, 7)}
    acoes = [(-1, 0), (1, 0), (0, -1), (0, 1)]
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
            q(s)[a] += alpha * (r + gamma * q(s2).max() * (not fim) - q(s)[a]); modelo[(s, a)] = (r, s2, fim)
            chaves = list(modelo.keys())
            for _ in range(n_plan):
                (ps, pa) = chaves[rng.integers(len(chaves))]; pr, ps2, pf = modelo[(ps, pa)]
                q(ps)[pa] += alpha * (pr + gamma * q(ps2).max() * (not pf) - q(ps)[pa])
            s = s2
        passos.append(n)
    return np.array(passos)
# ---------- modelo aprendido (CartPole) + MPC
def coletar(n, semente=0):
    env = gym.make("CartPole-v1"); rng = np.random.default_rng(semente); O = []; A = []; O2 = []; o, _ = env.reset(seed=semente)
    for _ in range(n):
        a = int(rng.integers(2)); o2, r, te, tr, _ = env.step(a); O.append(o); A.append(a); O2.append(o2); o = o2
        if te or tr: o, _ = env.reset()
    return np.array(O), np.array(A), np.array(O2)
def aprender_modelo(O, A, O2, n_arv=50):
    return ExtraTreesRegressor(n_estimators=n_arv, min_samples_leaf=2, random_state=0, n_jobs=1).fit(np.column_stack([O, A]), O2 - O)
def terminou(s): return (np.abs(s[:, 0]) > 2.4) | (np.abs(s[:, 2]) > 0.2095)
def planejar(modelo, s0, rng, H=15, N=200):
    seqs = rng.integers(0, 2, (N, H)); s = np.tile(s0, (N, 1)); vivo = np.ones(N, bool); ret = np.zeros(N)
    for h in range(H):
        s = s + modelo.predict(np.column_stack([s, seqs[:, h]])); ret += vivo; vivo &= ~terminou(s)
    return int(seqs[np.argmax(ret), 0])
def avaliar_mpc(modelo, n_ep=3, semente=700, **kw):
    env = gym.make("CartPole-v1"); rng = np.random.default_rng(semente); R = []
    for ep in range(n_ep):
        o, _ = env.reset(seed=semente + ep); tot = 0; fim = False
        while not fim:
            o, r, te, tr, _ = env.step(planejar(modelo, o, rng, **kw)); tot += r; fim = te or tr
        R.append(tot)
    return np.array(R)
# ---------- MCTS (velha)
LINHAS = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
def vencedor(t):
    for a, b, c in LINHAS:
        if t[a] != 0 and t[a] == t[b] == t[c]: return t[a]
    return 0 if 0 in t else 3
def jogadas(t): return [i for i in range(9) if t[i] == 0]
class No:
    def __init__(self, t, jogador, pai=None, mov=None):
        self.t, self.j, self.pai, self.mov = t, jogador, pai, mov; self.filhos = []; self.nao_exp = jogadas(t) if vencedor(t) == 0 else []; self.n = 0; self.w = 0.0
def uct(no, c=1.4): return max(no.filhos, key=lambda f: f.w / f.n + c * math.sqrt(math.log(no.n) / f.n))
def mcts(t, jogador, n_sim=500):
    raiz = No(t, jogador)
    for _ in range(n_sim):
        no = raiz
        while not no.nao_exp and no.filhos: no = uct(no)
        if no.nao_exp:
            m = no.nao_exp.pop(random.randrange(len(no.nao_exp))); t2 = list(no.t); t2[m] = no.j
            f = No(tuple(t2), 3 - no.j, no, m); no.filhos.append(f); no = f
        tt = list(no.t); j = no.j; v = vencedor(tuple(tt))
        while v == 0:
            m = random.choice(jogadas(tt)); tt[m] = j; j = 3 - j; v = vencedor(tuple(tt))
        while no is not None:
            quem = 3 - no.j; no.n += 1; no.w += 0.5 if v == 3 else (1.0 if v == quem else 0.0); no = no.pai
    return max(raiz.filhos, key=lambda f: f.n).mov
def partida(mcts_primeiro, n_sim):
    t = [0] * 9; j = 1
    while vencedor(tuple(t)) == 0:
        m = mcts(tuple(t), j, n_sim) if (j == 1) == mcts_primeiro else random.choice(jogadas(t))
        t[m] = j; j = 3 - j
    return vencedor(tuple(t))
def placar(n_sim, n_jogos=40):
    v = e = d = 0
    for k in range(n_jogos):
        r = partida(k % 2 == 0, n_sim); meu = 1 if k % 2 == 0 else 2
        if r == 3: e += 1
        elif r == meu: v += 1
        else: d += 1
    return v, e, d
