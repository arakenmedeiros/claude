"""
Livro: Aprendizado por Reforço — Processos de Decisão de Markov
Autor do livro: Araken de Medeiros Santos

Os blocos abaixo seguem a ordem do texto e devem ser executados em sequência (como células de um notebook).
Os trechos marcados como 'não executado' (por exemplo, downloads do Kaggle) aparecem comentados.
"""
import warnings; warnings.filterwarnings('ignore')

# ======================================================================
# Bloco 1
# ======================================================================
import itertools
import numpy as np

estados = ["alta", "baixa"]
acoes = ["buscar", "esperar", "recarregar"]
P = np.zeros((2, 3, 2))        # P[s, a, s'] = probabilidade
R = np.zeros((2, 3))           # R[s, a]     = recompensa esperada

P[0, 0] = [0.7, 0.3]; R[0, 0] = 4                 # alta, buscar
P[0, 1] = [1.0, 0.0]; R[0, 1] = 1                 # alta, esperar
P[0, 2] = [1.0, 0.0]; R[0, 2] = 0                 # alta, recarregar (inútil)
P[1, 0] = [0.4, 0.6]; R[1, 0] = 0.4 * (-3) + 0.6 * 4   # baixa, buscar
P[1, 1] = [0.0, 1.0]; R[1, 1] = 1                 # baixa, esperar
P[1, 2] = [1.0, 0.0]; R[1, 2] = 0                 # baixa, recarregar
gamma = 0.9

def avaliar(politica):
    """Resolve v = (I - γ P_π)^(-1) r_π para uma política determinística."""
    P_pi = np.array([P[s, politica[s]] for s in range(2)])
    r_pi = np.array([R[s, politica[s]] for s in range(2)])
    return np.linalg.solve(np.eye(2) - gamma * P_pi, r_pi)

print("política (alta, baixa)      | v(alta)  v(baixa)")
for pol in itertools.product(range(3), repeat=2):
    v = avaliar(pol)
    print(f"{acoes[pol[0]]:10s} {acoes[pol[1]]:10s} | {v[0]:7.2f} {v[1]:8.2f}")

# ======================================================================
# Bloco 2
# ======================================================================
import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=True)
Pf = env.unwrapped.P                    # Pf[s][a] = lista de (prob, próximo_estado, recompensa, terminou)
nS, nA = env.observation_space.n, env.action_space.n
print("Estados:", nS, "| Ações:", nA)
print("Do estado 14, ação 2 (direita):", Pf[14][2])

# converte para arrays P[s,a,s'] e R[s,a]
Pm = np.zeros((nS, nA, nS)); Rm = np.zeros((nS, nA))
for s in range(nS):
    for a in range(nA):
        for prob, s2, r, fim in Pf[s][a]:
            Pm[s, a, s2] += prob
            Rm[s, a] += prob * r

# política aleatória: cada ação com probabilidade 1/4
pol = np.full((nS, nA), 1 / nA)
P_pi = np.einsum("sa,sat->st", pol, Pm)    # média das transições sob a política
r_pi = (pol * Rm).sum(axis=1)
v = np.linalg.solve(np.eye(nS) - 0.99 * P_pi, r_pi)
print("Valor de cada estado sob a política aleatória:")
print(v.reshape(4, 4).round(3))

# ======================================================================
# Bloco 3
# ======================================================================
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = fetch_ucirepo(id=352).data.original.copy()      # Online Retail, UCI
df = df.dropna(subset=["CustomerID"])
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)
        & (~df["InvoiceNo"].astype(str).str.startswith("C"))]   # tira cancelamentos
df["mes"] = pd.to_datetime(df["InvoiceDate"]).dt.to_period("M")
df["gasto"] = df["Quantity"] * df["UnitPrice"]
gasto = df.pivot_table(index="CustomerID", columns="mes", values="gasto",
                       aggfunc="sum", fill_value=0.0)          # cliente × mês

G = gasto.to_numpy(); n, M = G.shape
comprou = G > 0
primeira = comprou.argmax(axis=1)                              # mês da primeira compra
ultima = np.maximum.accumulate(np.where(comprou, np.arange(M), -1), axis=1)

linhas = []
for j in range(2, M):
    ativo = primeira <= j                                      # só quem já era cliente
    linhas.append(pd.DataFrame({
        "cliente": gasto.index[ativo], "t": j,
        "rec": np.minimum(j - ultima[ativo, j], 6),           # recência (máx. 6 meses)
        "freq": comprou[ativo, j-2:j+1].sum(axis=1),          # compras nos últimos 3 meses
        "mon": np.log1p(G[ativo, j-2:j+1].sum(axis=1)),       # gasto (escala log)
        "gasto_prox": G[ativo, j+1] if j + 1 < M else np.nan}))
feat = pd.concat(linhas, ignore_index=True)

Z = StandardScaler().fit_transform(feat[["rec", "freq", "mon"]])
feat["estado"] = KMeans(n_clusters=5, n_init=10, random_state=0).fit_predict(Z)
ordem = feat.groupby("estado")["mon"].mean().sort_values().index   # numera do menor ao maior gasto
feat["estado"] = feat["estado"].map({e: i for i, e in enumerate(ordem)})
print(feat.groupby("estado")[["rec", "freq", "mon"]].mean().round(2))

feat = feat.sort_values(["cliente", "t"])
feat["prox"] = feat.groupby("cliente")["estado"].shift(-1)         # estado no mês seguinte
tr = feat.dropna(subset=["prox"])
C = pd.crosstab(tr["estado"], tr["prox"]).reindex(
        index=range(5), columns=range(5), fill_value=0).to_numpy().astype(float)
P_est = C / C.sum(axis=1, keepdims=True)                           # cada linha soma 1
r_est = tr.groupby("estado")["gasto_prox"].mean().reindex(range(5)).to_numpy()

v_clv = np.linalg.solve(np.eye(5) - 0.95 * P_est, r_est)           # Bellman matricial
print("Matriz de transição estimada:"); print(P_est.round(2))
print("Gasto médio no mês seguinte:", r_est.round(1))
print("Valor de vida de cada estado (γ = 0,95):", v_clv.round(0))

np.save("P_clientes.npy", P_est); np.save("r_clientes.npy", r_est)   # serão usados nos Capítulos 4 e 5
feat.to_csv("painel_clientes.csv", index=False)                       # painel cliente-mês: será usado no Capítulo 7

# ======================================================================
# Bloco 4
# ======================================================================
import numpy as np

# --- Parte 1: atualização de crença (a porta do tesouro) ---
# Estado oculto: o tesouro está atrás da porta ESQUERDA (0) ou DIREITA (1). O estado não muda.
# Observação: um sensor "ouve" o lado certo com 85% de acerto.
precisao = 0.85

def atualizar_crenca(crenca, obs):
    """Regra de Bayes: nova crença ∝ P(obs | estado) × crença antiga."""
    verossimilhanca = np.where(np.arange(2) == obs, precisao, 1 - precisao)
    nova = verossimilhanca * crenca
    return nova / nova.sum()

crenca = np.array([0.5, 0.5])
print("crença inicial (esquerda, direita):", crenca.round(3))
for obs, nome in [(0, "ouviu esquerda"), (0, "ouviu esquerda"), (1, "ouviu direita"), (0, "ouviu esquerda")]:
    crenca = atualizar_crenca(crenca, obs)
    print(f"  após {nome:14s} -> crença = {crenca.round(3)}")

# --- Parte 2: quando uma observação isolada não basta (o T-labirinto) ---
# No primeiro passo, um sinal mostra o lado do prêmio (0 ou 1). Depois, o agente anda por um corredor
# (observação 2, sem informação) e, no cruzamento, escolhe esquerda (0) ou direita (1).
CORREDOR = 2

def episodio(politica, rng, tam_corredor=5):
    sinal = int(rng.integers(2)); memoria = None
    for t in range(tam_corredor + 1):
        obs = sinal if t == 0 else CORREDOR
        acao, memoria = politica(obs, memoria)
    return float(acao == sinal)                       # a última ação é a escolha no cruzamento

def sem_memoria(obs, memoria):                        # decide só com a observação atual
    return (obs if obs != CORREDOR else 0), None

def com_memoria(obs, memoria):                        # guarda o último sinal visto
    memoria = obs if obs != CORREDOR else memoria
    return memoria, memoria

rng = np.random.default_rng(0)
for nome, pol in [("sem memória (só a observação atual)", sem_memoria), ("com memória (guarda o sinal)", com_memoria)]:
    taxa = np.mean([episodio(pol, rng) for _ in range(2000)])
    print(f"{nome:36s} | taxa de acerto = {taxa:.3f}")

# ======================================================================
# Bloco 5
# ======================================================================
from sklearn.metrics import silhouette_score

amostra = np.random.default_rng(0).choice(len(Z), size=min(5000, len(Z)), replace=False)
for k in range(3, 9):
    rotulos = KMeans(n_clusters=k, n_init=5, random_state=0).fit_predict(Z)
    print(f"k = {k} | silhueta = {silhouette_score(Z[amostra], rotulos[amostra]):.3f}")

for g in [0.8, 0.9, 0.95]:
    v = np.linalg.solve(np.eye(5) - g * P_est, r_est)
    print(f"γ = {g} | valor do melhor estado = {v.max():.0f} | resíduo de Bellman = "
          f"{np.abs(v - (r_est + g * P_est @ v)).max():.2e}")

