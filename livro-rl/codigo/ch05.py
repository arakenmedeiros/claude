"""
Livro: Aprendizado por Reforço — Métodos de Monte Carlo
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
from collections import defaultdict
from sklearn.utils import resample

env = gym.make("Blackjack-v1", sab=True)

def gerar_episodio(politica):
    """Joga um episódio e devolve a lista de (estado, ação, recompensa)."""
    obs, _ = env.reset(); ep, fim = [], False
    while not fim:
        a = politica(obs)
        obs2, r, term, trunc, _ = env.step(a)
        ep.append((obs, a, r)); obs = obs2; fim = term or trunc
    return ep

def retornos_do_episodio(ep, gamma=1.0):
    """Retorno G_t de cada passo, calculado de trás para frente."""
    G, ret = 0.0, [0.0] * len(ep)
    for t in range(len(ep) - 1, -1, -1):
        G = ep[t][2] + gamma * G; ret[t] = G
    return ret

def parar_em_20(obs):                       # 0 = parar, 1 = pedir carta
    return 0 if obs[0] >= 20 else 1

def mc_predicao(politica, n_episodios):
    soma, cont = defaultdict(float), defaultdict(int)
    for _ in range(n_episodios):
        ep = gerar_episodio(politica); ret = retornos_do_episodio(ep); vistos = set()
        for t, (s, a, r) in enumerate(ep):
            if s not in vistos:             # primeira visita
                vistos.add(s); soma[s] += ret[t]; cont[s] += 1
    return {s: soma[s] / cont[s] for s in soma}

V = mc_predicao(parar_em_20, 200_000)
print(f"Estados estimados: {len(V)}")
for s in [(21, 10, False), (20, 10, False), (16, 10, False), (12, 5, False)]:
    print(f"v{s} = {V[s]:+.2f}")

# retorno médio da política, com intervalo de confiança por bootstrap (scikit-learn)
r_ep = np.array([gerar_episodio(parar_em_20)[-1][2] for _ in range(20_000)])
boot = [resample(r_ep, random_state=b).mean() for b in range(300)]
print(f"Retorno médio de 'parar em 20': {r_ep.mean():+.3f} (IC 95%: {np.percentile(boot, 2.5):+.3f} a {np.percentile(boot, 97.5):+.3f})")

# ======================================================================
# Bloco 2
# ======================================================================
rng = np.random.default_rng(0)
Q = defaultdict(lambda: np.zeros(2))        # Q[estado] = [valor de parar, valor de pedir]
N = defaultdict(lambda: np.zeros(2))

def politica_eps(eps):
    def p(obs):
        if rng.random() < eps: return int(rng.integers(2))      # explora
        q = Q[obs]
        return int(q.argmax()) if q[0] != q[1] else int(rng.integers(2))   # aproveita
    return p

def avaliar(politica, n=20_000):
    return np.mean([gerar_episodio(politica)[-1][2] for _ in range(n)])

n_total = 200_000
for k in range(n_total):
    eps = max(0.05, 0.5 * (1 - k / n_total))               # ε diminui ao longo do treino
    ep = gerar_episodio(politica_eps(eps)); ret = retornos_do_episodio(ep); vistos = set()
    for t, (s, a, r) in enumerate(ep):
        if (s, a) not in vistos:                             # primeira visita ao par
            vistos.add((s, a)); N[s][a] += 1
            Q[s][a] += (ret[t] - Q[s][a]) / N[s][a]          # média incremental
    if (k + 1) in (1_000, 10_000, 50_000, 200_000):
        print(f"após {k+1:7d} episódios | retorno médio da política gulosa = {avaliar(politica_eps(0.0)):+.3f}")
print("Referência: parar só em 20 →", f"{r_ep.mean():+.3f}")

# ======================================================================
# Bloco 3
# ======================================================================
from sklearn.tree import DecisionTreeClassifier, export_text

estados = [s for s in Q if N[s].sum() >= 20]               # estados com experiência suficiente
X_pol = np.array([[s[0], s[1], int(s[2])] for s in estados])
y_pol = np.array([int(Q[s].argmax()) for s in estados])    # 0 = parar, 1 = pedir
arvore_pol = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_pol, y_pol)

print("Fidelidade da árvore à política:", round(arvore_pol.score(X_pol, y_pol), 3))
print(export_text(arvore_pol, feature_names=["soma", "crupie", "as_usavel"]))

# ======================================================================
# Bloco 4
# ======================================================================
def estimar_is(n_episodios, semente=1):
    rng_l = np.random.default_rng(semente); rho, G = [], []
    for _ in range(n_episodios):
        obs, _ = env.reset(); fim, r_, ret = False, 1.0, 0.0
        while not fim:
            a = int(rng_l.integers(2))                          # comportamento: aleatório
            r_ *= 2.0 if a == parar_em_20(obs) else 0.0          # π(a|s)/b(a|s) = 1/0,5 ou 0/0,5
            obs, r, term, trunc, _ = env.step(a); ret += r; fim = term or trunc
        rho.append(r_); G.append(ret)
    rho, G = np.array(rho), np.array(G)
    return (rho * G).mean(), (rho * G).sum() / max(rho.sum(), 1e-12), (rho > 0).mean()

for n in [1_000, 10_000, 50_000]:
    ordinaria, ponderada, uteis = estimar_is(n)
    print(f"n = {n:6d} | ordinária = {ordinaria:+.3f} | ponderada = {ponderada:+.3f} | episódios úteis = {uteis:.1%}")
print(f"Valor verdadeiro (simulando a política-alvo): {r_ep.mean():+.3f}")

# ======================================================================
# Bloco 5
# ======================================================================
from sklearn.utils import resample

P_cli = np.load("P_clientes.npy"); r_cli = np.load("r_clientes.npy"); K = len(r_cli)
gamma = 0.95; v_exato = np.linalg.solve(np.eye(K) - gamma * P_cli, r_cli)

def retorno_simulado(s0, rng_l, horizonte=200):
    s, G, desc = s0, 0.0, 1.0
    for _ in range(horizonte):
        G += desc * r_cli[s]; desc *= gamma
        s = rng_l.choice(K, p=P_cli[s])
    return G

rng_c = np.random.default_rng(0)
for s0 in range(K):
    G = np.array([retorno_simulado(s0, rng_c) for _ in range(300)])
    boot = [resample(G, random_state=b).mean() for b in range(200)]
    print(f"estado {s0}: MC = {G.mean():8.1f} (IC 95%: {np.percentile(boot, 2.5):.1f} a {np.percentile(boot, 97.5):.1f}) | exato = {v_exato[s0]:.1f}")

