"""
Livro: Aprendizado por Reforço — Aproximação de Funções
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

def baird(passos=3000, alpha=0.01, gamma=0.99, semente=0):
    rng = np.random.default_rng(semente)
    X = np.zeros((7, 8))                                  # características de cada estado
    for i in range(6): X[i, i] = 2; X[i, 7] = 1
    X[6, 6] = 1; X[6, 7] = 2
    w = np.array([1, 1, 1, 1, 1, 1, 10, 1], float); s = int(rng.integers(7)); hist = [w.copy()]
    for _ in range(passos):
        if rng.random() < 6 / 7: s2, rho = int(rng.integers(6)), 0.0     # comportamento: "tracejada"
        else:                    s2, rho = 6, 7.0                         # "sólida" (política-alvo): ρ = 1/(1/7)
        delta = 0 + gamma * X[s2] @ w - X[s] @ w                          # recompensa zero
        w += alpha * rho * delta * X[s]; s = s2; hist.append(w.copy())
    return np.array(hist)

h = baird()
for t in [0, 500, 1000, 2000, 3000]:
    print(f"passo {t:4d} | maior |w| = {np.abs(h[t]).max():10.1f}")

# ======================================================================
# Bloco 2
# ======================================================================
env = gym.make("MountainCar-v0"); rng = np.random.default_rng(0)
N_T, N_B = 8, 8                                   # 8 grades; 8 divisões por variável
baixo, alto = np.array([-1.2, -0.07]), np.array([0.6, 0.07])

def caracteristicas(o):
    """Índices dos quadrados acesos (um por grade)."""
    z = (o - baixo) / (alto - baixo) * N_B
    idx = []
    for t in range(N_T):
        desloc = t / N_T * np.array([1, 3])       # deslocamento diferente em cada grade
        c = np.floor(z + desloc).astype(int).clip(0, N_B)
        idx.append(t * (N_B + 1) ** 2 + c[0] * (N_B + 1) + c[1])
    return idx

W = np.zeros((3, N_T * (N_B + 1) ** 2))           # um vetor de pesos por ação
q = lambda idx, a: W[a, idx].sum()                # q̂ = soma dos pesos dos quadrados acesos
alpha = 0.5 / N_T; passos_ep = []

for ep in range(500):
    o, _ = env.reset(seed=int(rng.integers(1_000_000))); idx = caracteristicas(o)
    a = int(np.argmax([q(idx, b) for b in range(3)])); n, fim = 0, False
    while not fim:
        o2, r, term, trunc, _ = env.step(a); fim = term or trunc; n += 1
        idx2 = caracteristicas(o2)
        if term:
            W[a, idx] += alpha * (r - q(idx, a)); break
        a2 = int(np.argmax([q(idx2, b) for b in range(3)]))      # valores iniciais (0) otimistas: sem ε
        W[a, idx] += alpha * (r + q(idx2, a2) - q(idx, a))        # semi-gradiente SARSA
        idx, a = idx2, a2
    passos_ep.append(n)
    if (ep + 1) % 100 == 0:
        print(f"episódios {ep-98:3d} a {ep+1:3d} | passos médios até a meta = {np.mean(passos_ep[-100:]):6.1f}")

# ======================================================================
# Bloco 3
# ======================================================================
from sklearn.preprocessing import KBinsDiscretizer, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

r = np.random.default_rng(2)
v_verd = lambda x: np.sin(2.5 * np.pi * x) * 0.8 * np.exp(-x) + 0.4 * x
x_tr = np.sort(r.uniform(0, 1, 40)).reshape(-1, 1); y_tr = v_verd(x_tr.ravel()) + r.normal(0, 0.12, 40)
x_te = np.linspace(0, 1, 400).reshape(-1, 1); y_te = v_verd(x_te.ravel())

faixas = KBinsDiscretizer(n_bins=6, encode="ordinal", strategy="uniform").fit(x_tr)
idx_tr = faixas.transform(x_tr).ravel().astype(int)
media_faixa = np.array([y_tr[idx_tr == k].mean() for k in range(6)])
modelos = {
    "tabela (6 faixas)": lambda X: media_faixa[faixas.transform(X).ravel().astype(int)],
    "reta":              LinearRegression().fit(x_tr, y_tr).predict,
    "polinômio grau 5":  make_pipeline(PolynomialFeatures(5), Ridge(alpha=1e-3)).fit(x_tr, y_tr).predict,
    "RBF + Ridge":       make_pipeline(RBFSampler(gamma=20, n_components=60, random_state=0), Ridge(alpha=1e-2)).fit(x_tr, y_tr).predict,
}
for nome, prever in modelos.items():
    print(f"{nome:18s} | erro quadrático médio no teste = {mean_squared_error(y_te, prever(x_te)):.4f}")

# ======================================================================
# Bloco 4
# ======================================================================
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor

env = gym.make("CartPole-v1"); rng = np.random.default_rng(0)
amostra, o = [], env.reset(seed=0)[0]
for _ in range(5000):
    o, r, term, trunc, _ = env.step(int(rng.integers(2))); amostra.append(o)
    if term or trunc: o = env.reset()[0]
amostra = np.array(amostra); esc = StandardScaler().fit(amostra)
rbf = RBFSampler(gamma=0.5, n_components=300, random_state=0).fit(esc.transform(amostra))
feat = lambda o: rbf.transform(esc.transform(np.atleast_2d(o)))

modelos_q = [SGDRegressor(learning_rate="constant", eta0=0.005, alpha=1e-6, random_state=0) for _ in range(2)]
for m in modelos_q: m.partial_fit(feat(amostra[0]), [10.0])

retornos = []
for ep in range(400):
    eps = max(0.02, 0.5 * (1 - ep / 300)); o, _ = env.reset(seed=int(rng.integers(1_000_000))); x = feat(o); fim, total = False, 0
    while not fim:
        a = int(rng.integers(2)) if rng.random() < eps else int(np.argmax([m.predict(x)[0] for m in modelos_q]))
        o2, r, term, trunc, _ = env.step(a); fim = term or trunc; total += r; x2 = feat(o2)
        alvo = r if term else r + 0.99 * max(m.predict(x2)[0] for m in modelos_q)    # bootstrapping + off-policy
        modelos_q[a].partial_fit(x, [alvo]); x = x2
    retornos.append(total)
print(f"Retorno médio nos últimos 100 episódios: {np.mean(retornos[-100:]):.1f}  (um agente aleatório faz ≈ 22)")

# ======================================================================
# Bloco 5
# ======================================================================
from sklearn.ensemble import ExtraTreesRegressor

O, A, R, O2, T = [], [], [], [], []
o = env.reset(seed=0)[0]
for _ in range(20_000):                               # 20.000 transições de jogo aleatório
    a = int(rng.integers(2)); o2, r, term, trunc, _ = env.step(a)
    O.append(o); A.append(a); R.append(r); O2.append(o2); T.append(term); o = o2
    if term or trunc: o = env.reset()[0]
O, A, R, O2, T = map(np.array, (O, A, R, O2, T)); X = np.column_stack([O, A])

def q_max(modelo, O2):                                 # max_a' q(s', a') em lote
    n = len(O2)
    Xb = np.vstack([np.column_stack([O2, np.zeros(n)]), np.column_stack([O2, np.ones(n)])])
    q = modelo.predict(Xb); return np.maximum(q[:n], q[n:])

def fqi(fabrica, n_iter=20, gamma=0.99):
    modelo = None
    for _ in range(n_iter):
        y = R.astype(float) if modelo is None else R + gamma * q_max(modelo, O2) * (~T)
        modelo = fabrica().fit(X, y)
    return modelo

def avaliar(modelo, n=20):
    ret = []
    for ep in range(n):
        o, _ = env.reset(seed=3000 + ep); total, fim = 0, False
        while not fim:
            q = modelo.predict(np.array([[*o, 0], [*o, 1]]))
            o, r, term, trunc, _ = env.step(int(q.argmax())); total += r; fim = term or trunc
        ret.append(total)
    return np.mean(ret), min(ret), max(ret)

m_arv = fqi(lambda: ExtraTreesRegressor(n_estimators=30, min_samples_leaf=5, random_state=0))
m_rbf = fqi(lambda: make_pipeline(StandardScaler(), RBFSampler(gamma=0.3, n_components=400, random_state=0), Ridge(alpha=1.0)))
for nome, m in [("ExtraTrees", m_arv), ("RBF + Ridge", m_rbf)]:
    media, mn, mx = avaliar(m)
    print(f"{nome:12s} | retorno médio = {media:6.1f} (mín {mn:.0f}, máx {mx:.0f})")

# ======================================================================
# Bloco 6
# ======================================================================
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor

painel = pd.read_csv("painel_clientes.csv").sort_values(["cliente", "t"])
for c in ["rec", "freq", "mon"]:
    painel[c + "_prox"] = painel.groupby("cliente")[c].shift(-1)
d = painel.dropna(subset=["rec_prox", "gasto_prox"]).copy()
X0 = d[["rec", "freq", "mon"]].to_numpy(); X1 = d[["rec_prox", "freq_prox", "mon_prox"]].to_numpy()
r_ = d["gasto_prox"].to_numpy(); grupos = d["cliente"].to_numpy()

def fvi(fabrica, gamma=0.9, n_iter=15):
    V = None; escalas = []
    for _ in range(n_iter):
        y = r_ if V is None else r_ + gamma * V.predict(X1)
        V = fabrica().fit(X0, y); escalas.append(np.abs(V.predict(X0)).mean())
    return V, escalas

fab_hgb = lambda: HistGradientBoostingRegressor(max_iter=100, learning_rate=0.1, random_state=0)
treino, teste = next(GroupKFold(n_splits=5).split(X0, groups=grupos))
V_hgb, esc_hgb = fvi(fab_hgb)
print("Escala média de |V| a cada iteração:", np.round(esc_hgb, 1))
erro_td = ((V_hgb.predict(X0[teste]) - (r_[teste] + 0.9 * V_hgb.predict(X1[teste]))) ** 2).mean()
print("Erro TD (MSE) em clientes de teste (modelo treinado em todos os dados):", round(erro_td, 1))

