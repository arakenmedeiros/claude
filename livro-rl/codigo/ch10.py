"""
Livro: Aprendizado por Reforço — Algoritmos Modernos: PPO, DDPG, TD3 e SAC
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
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

envs = make_vec_env("CartPole-v1", n_envs=4, seed=0)          # 4 ambientes em paralelo
ppo = PPO("MlpPolicy", envs, n_steps=256, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
          clip_range=0.2, ent_coef=0.0, learning_rate=3e-4, seed=0, verbose=0)
ppo.learn(total_timesteps=40_000)
media, desvio = evaluate_policy(ppo, gym.make("CartPole-v1"), n_eval_episodes=20)
print(f"PPO no CartPole: retorno médio = {media:.1f} ± {desvio:.1f}")

# ======================================================================
# Bloco 2
# ======================================================================
import torch

def objetivo_ppo(razao, vantagem, eps=0.2):
    r = torch.as_tensor(razao, dtype=torch.float32); A = torch.as_tensor(vantagem, dtype=torch.float32)
    return torch.minimum(r * A, torch.clamp(r, 1 - eps, 1 + eps) * A)

razoes = [0.5, 0.9, 1.0, 1.1, 1.5]
print("razão r        :", razoes)
print("A = +1 (boa)   :", objetivo_ppo(razoes, [1.0] * 5).numpy().round(2))     # satura em 1,2
print("A = −1 (ruim)  :", objetivo_ppo(razoes, [-1.0] * 5).numpy().round(2))    # satura em −0,8 do lado esquerdo

# ======================================================================
# Bloco 3
# ======================================================================
import time
amb = gym.make("Pendulum-v1")
resultados = {}
for nome, cls, kw, passos in [
        ("SAC", SAC, dict(learning_starts=500), 10_000),
        ("TD3", TD3, dict(learning_starts=500), 10_000),
        ("PPO", PPO, dict(n_steps=1024, batch_size=64, gae_lambda=0.95, gamma=0.9, n_epochs=10,
                          learning_rate=1e-3, use_sde=True, sde_sample_freq=4), 50_000)]:
    t0 = time.time(); m = cls("MlpPolicy", amb, seed=0, verbose=0, **kw); m.learn(passos)
    media, desvio = evaluate_policy(m, gym.make("Pendulum-v1"), n_eval_episodes=10)
    resultados[nome] = media
    print(f"{nome:4s} | {passos:6d} passos | retorno médio = {media:8.1f} ± {desvio:5.1f} | {time.time()-t0:3.0f} s", flush=True)

# ======================================================================
# Bloco 4
# ======================================================================
from sklearn.model_selection import ParameterGrid
import pandas as pd

linhas = []
for params in ParameterGrid({"n_steps": [64, 256], "gamma": [0.9, 0.99]}):
    notas = []
    for s in range(2):
        e = make_vec_env("CartPole-v1", n_envs=4, seed=s)
        m = PPO("MlpPolicy", e, seed=s, verbose=0, batch_size=64, n_epochs=10, **params).learn(total_timesteps=20_000)
        notas.append(evaluate_policy(m, gym.make("CartPole-v1"), n_eval_episodes=10)[0])
    linhas.append({**params, "media": np.mean(notas), "min": np.min(notas), "max": np.max(notas)})
print(pd.DataFrame(linhas).round(1).to_string(index=False))

# ======================================================================
# Bloco 5
# ======================================================================
import pandas as pd
from ucimlrepo import fetch_ucirepo
from gymnasium import spaces
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error

vinho = fetch_ucirepo(id=186)
X_all = vinho.data.features; y_all = vinho.data.targets.iloc[:, 0].astype(float)
cols = [c for c in ["alcohol", "volatile_acidity", "sulphates", "pH", "citric_acid"] if c in X_all.columns]
Xv = X_all[cols].to_numpy(float); lo, hi = Xv.min(0), Xv.max(0); Xn = (Xv - lo) / (hi - lo)
Xtr, Xte, ytr, yte = train_test_split(Xn, y_all.to_numpy(), test_size=0.25, random_state=0)
simulador = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=0, n_jobs=-1).fit(Xtr, ytr)
print("variáveis:", cols, "| erro absoluto médio no teste:", round(mean_absolute_error(yte, simulador.predict(Xte)), 3))
vizinhos = NearestNeighbors(n_neighbors=1).fit(Xn)

class ProcessoEnv(gym.Env):
    def __init__(self, passos=15):
        super().__init__(); self.passos = passos; k = len(cols)
        self.action_space = spaces.Box(-1.0, 1.0, (k,), np.float32); self.observation_space = spaces.Box(0.0, 1.0, (k,), np.float32)
    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self.x = Xn[self.np_random.integers(len(Xn))].astype(np.float32).copy(); self.t = 0
        return self.x.copy(), {}
    def step(self, acao):
        ajuste = 0.1 * np.clip(acao, -1, 1); self.x = np.clip(self.x + ajuste, 0.0, 1.0).astype(np.float32); self.t += 1
        qualidade = float(simulador.predict(self.x.reshape(1, -1))[0])
        return self.x.copy(), qualidade - 0.5 * float((ajuste ** 2).sum()) * 100, False, self.t >= self.passos, {}

env_p = ProcessoEnv(); agente = SAC("MlpPolicy", env_p, learning_starts=300, seed=0, verbose=0).learn(total_timesteps=3_000)
obs, _ = env_p.reset(seed=1); fim = False
while not fim:
    obs, r, term, trunc, _ = env_p.step(agente.predict(obs, deterministic=True)[0]); fim = term or trunc
dist = vizinhos.kneighbors(obs.reshape(1, -1))[0][0, 0]
print("estado final:", obs.round(2), "| qualidade prevista:", round(float(simulador.predict(obs.reshape(1, -1))[0]), 2),
      "| distância ao dado real mais próximo:", round(float(dist), 3))

