"""
Livro: Aprendizado por Reforço — Uma plataforma de ofertas para reter clientes
Autor do livro: Araken de Medeiros Santos

Os blocos abaixo seguem a ordem do texto e devem ser executados em sequência (como células de um notebook).
Os trechos marcados como 'não executado' (por exemplo, downloads do Kaggle) aparecem comentados.
"""
import warnings; warnings.filterwarnings('ignore')

# ======================================================================
# Bloco 1
# ======================================================================
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold

painel = pd.read_csv("painel_clientes.csv").sort_values(["cliente", "t"])
for c in ["rec", "freq", "mon"]:
    painel[c + "_prox"] = painel.groupby("cliente")[c].shift(-1)
d = painel.dropna(subset=["rec_prox", "gasto_prox"]).copy()
X = d[["rec", "freq", "mon"]].to_numpy(); Y = d[["rec_prox", "freq_prox", "mon_prox", "gasto_prox"]].to_numpy()
grupos = d["cliente"].to_numpy(); idx_tr, idx_te = next(GroupKFold(n_splits=3).split(X, groups=grupos))
mundo = ExtraTreesRegressor(n_estimators=40, min_samples_leaf=5, random_state=0, n_jobs=1).fit(X[idx_tr], Y[idx_tr])
residuos = Y[idx_tr] - mundo.predict(X[idx_tr])

class ClienteEnv(gym.Env):
    """Um cliente por episódio. Ações: 0 = nada, 1 = cupom 10%, 2 = cupom 20% (efeitos HIPOTÉTICOS)."""
    DESCONTO = np.array([0.0, 0.10, 0.20]); EFEITO_BASE = np.array([0.0, 0.40, 0.60]); CUSTO_FIXO = 1.0
    def __init__(self, X_inicio, efeito_escala=1.0, horizonte=6, orcamento=3, semente=0):
        super().__init__(); self.Xi, self.esc, self.H, self.orc = X_inicio, efeito_escala, horizonte, orcamento
        self.action_space = spaces.Discrete(3); self.observation_space = spaces.Box(0.0, 10.0, (4,), np.float32)
        self.rng = np.random.default_rng(semente)
    def _obs(self): return np.array([self.s[0] / 6 * 10, min(self.s[1], 3) / 3 * 10, min(self.s[2], 10) / 10 * 10, self.t / self.H * 10], np.float32)
    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self.s = self.Xi[self.np_random.integers(len(self.Xi))].astype(float).copy(); self.t = 0; self.usados = 0
        return self._obs(), {}
    def step(self, a):
        p = mundo.predict(self.s.reshape(1, -1))[0] + residuos[self.np_random.integers(len(residuos))]
        u = self.esc * self.EFEITO_BASE[a] * (0.3 + 0.7 * min(self.s[0], 2) / 2)            # efeito maior em quem está "esfriando"
        gasto = max(p[3], 0.0) * (1 + u); r = gasto * (1 - self.DESCONTO[a]) - (self.CUSTO_FIXO if a > 0 else 0.0)
        self.usados += int(a > 0); r -= 5.0 * max(0, self.usados - self.orc) * int(a > 0)  # penalidade acima do orçamento
        p[0] = p[0] * (1 - min(0.9, 2.0 * u))                                             # a oferta também "reativa" o cliente (reduz a recência)
        self.s = np.array([max(p[0], 0), max(p[1], 0), max(p[2], 0)]); self.t += 1
        return self._obs(), float(r), False, self.t >= self.H, {}

env_tr = ClienteEnv(X[idx_tr]); env_te = ClienteEnv(X[idx_te], semente=1)
print("estado inicial (treino):", env_tr.reset(seed=0)[0].round(2))

# ======================================================================
# Bloco 2
# ======================================================================
from stable_baselines3 import DQN, PPO

def avaliar_politica(env, politica, n=200, semente=0):
    ret = []
    for i in range(n):
        obs, _ = env.reset(seed=semente + i); total, fim = 0.0, False
        while not fim:
            obs, r, term, trunc, _ = env.step(int(politica(obs))); total += r; fim = term or trunc
        ret.append(total)
    return np.array(ret)

nunca = lambda obs: 0
regra = lambda obs: 2 if obs[0] / 10 * 6 >= 2 else (1 if obs[0] / 10 * 6 >= 1 else 0)
agente = DQN("MlpPolicy", env_tr, learning_rate=5e-4, buffer_size=30_000, learning_starts=500, batch_size=64,
             target_update_interval=250, exploration_fraction=0.3, policy_kwargs=dict(net_arch=[64, 64]), verbose=0, seed=0)
agente.learn(total_timesteps=5_000)
for nome, pol in [("nunca ofertar", nunca), ("regra RFM", regra), ("DQN", lambda o: agente.predict(o, deterministic=True)[0])]:
    r = avaliar_politica(env_te, pol, n=50)
    print(f"{nome:14s} | receita líquida por cliente em 6 meses: média = {r.mean():7.1f} (EP = {r.std() / np.sqrt(len(r)):.1f})")

# ======================================================================
# Bloco 3
# ======================================================================
from sklearn.model_selection import ParameterGrid

linhas = []
for p in ParameterGrid({"efeito_escala": [0.0, 0.5, 1.0, 2.0]}):
    amb = ClienteEnv(X[idx_te], efeito_escala=p["efeito_escala"], semente=3)
    for nome, pol in [("nunca", nunca), ("regra RFM", regra), ("DQN", lambda o: agente.predict(o, deterministic=True)[0])]:
        linhas.append({**p, "política": nome, "receita": avaliar_politica(amb, pol, n=40).mean()})
print(pd.DataFrame(linhas).pivot(index="efeito_escala", columns="política", values="receita").round(1))

