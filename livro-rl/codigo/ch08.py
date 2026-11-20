"""
Livro: Aprendizado por Reforço — Deep Q-Networks e suas Extensões
Autor do livro: Araken de Medeiros Santos

Os blocos abaixo seguem a ordem do texto e devem ser executados em sequência (como células de um notebook).
Os trechos marcados como 'não executado' (por exemplo, downloads do Kaggle) aparecem comentados.
"""
import warnings; warnings.filterwarnings('ignore')

# ======================================================================
# Bloco 1
# ======================================================================
import random, time
from collections import deque
import numpy as np
import torch, torch.nn as nn
import gymnasium as gym
torch.set_num_threads(1)

def criar_rede(n_obs, n_acoes, ocultas=128):
    return nn.Sequential(nn.Linear(n_obs, ocultas), nn.ReLU(),
                         nn.Linear(ocultas, ocultas), nn.ReLU(),
                         nn.Linear(ocultas, n_acoes))

def treinar_dqn(semente=0, n_ep=300, usar_replay=True, usar_alvo=True, duplo=False, lr=5e-4, gamma=0.99,
                lote=64, tam_memoria=50_000, aquecimento=1000, troca_alvo=250, decaimento_eps=5000,
                env_id="CartPole-v1", verbose=False):
    random.seed(semente); np.random.seed(semente); torch.manual_seed(semente)
    env = gym.make(env_id); n_obs = env.observation_space.shape[0]; n_acoes = env.action_space.n
    env.action_space.seed(semente)                                 # o espaço de ações tem o seu próprio gerador aleatório
    q = criar_rede(n_obs, n_acoes); q_alvo = criar_rede(n_obs, n_acoes); q_alvo.load_state_dict(q.state_dict())
    if not usar_alvo: q_alvo = q                                   # sem rede-alvo: o alvo usa a própria rede
    if not usar_replay: tam_memoria, lote, aquecimento = 1, 1, 1   # sem memória: só a transição mais recente
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    memoria = deque(maxlen=tam_memoria); passo = 0; retornos = []
    for ep in range(n_ep):
        obs, _ = env.reset(seed=semente * 10_000 + ep); fim, total = False, 0.0
        while not fim:
            eps = max(0.05, 1 - passo / decaimento_eps)             # ε decresce linearmente
            if random.random() < eps: a = env.action_space.sample()
            else:
                with torch.no_grad(): a = int(q(torch.as_tensor(obs, dtype=torch.float32)).argmax())
            obs2, r, term, trunc, _ = env.step(a); fim = term or trunc; total += r
            memoria.append((obs, a, r, obs2, float(term))); obs = obs2; passo += 1
            if len(memoria) >= aquecimento:
                O, A, R, O2, T = map(np.array, zip(*random.sample(memoria, min(lote, len(memoria)))))
                O, O2, R, T = [torch.as_tensor(x, dtype=torch.float32) for x in (O, O2, R, T)]; A = torch.as_tensor(A)
                with torch.no_grad():
                    if duplo:                                        # Double DQN
                        a_star = q(O2).argmax(1, keepdim=True); proximo = q_alvo(O2).gather(1, a_star).squeeze(1)
                    else:
                        proximo = q_alvo(O2).max(1).values
                    y = R + gamma * (1 - T) * proximo                # alvo (sem valor futuro se terminou)
                pred = q(O).gather(1, A.unsqueeze(1)).squeeze(1)     # Q(s, a) das ações tomadas
                perda = nn.functional.smooth_l1_loss(pred, y)        # perda de Huber
                opt.zero_grad(); perda.backward(); nn.utils.clip_grad_norm_(q.parameters(), 10.0); opt.step()
            if usar_alvo and passo % troca_alvo == 0: q_alvo.load_state_dict(q.state_dict())
        retornos.append(total)
        if verbose and (ep + 1) % 50 == 0:
            print(f"episódio {ep+1:4d} | retorno médio (últimos 50) = {np.mean(retornos[-50:]):6.1f}", flush=True)
    return q, np.array(retornos)

t0 = time.time()
q_treinada, retornos = treinar_dqn(semente=0, n_ep=300, verbose=True)
print(f"Tempo de treino: {time.time() - t0:.0f} s")

# ======================================================================
# Bloco 2
# ======================================================================
from sklearn.utils import resample

def avaliar_rede(q, n=30, env_id="CartPole-v1"):
    env = gym.make(env_id); ret = []
    for i in range(n):
        obs, _ = env.reset(seed=50_000 + i); fim, total = False, 0.0
        while not fim:
            with torch.no_grad(): a = int(q(torch.as_tensor(obs, dtype=torch.float32)).argmax())
            obs, r, term, trunc, _ = env.step(a); total += r; fim = term or trunc
        ret.append(total)
    return np.array(ret)

ret_av = avaliar_rede(q_treinada)
boot = [resample(ret_av, random_state=b).mean() for b in range(300)]
print(f"Política gulosa: retorno médio = {ret_av.mean():.1f} (IC 95%: {np.percentile(boot, 2.5):.1f} a {np.percentile(boot, 97.5):.1f})")
print(f"Durante o treino, nos últimos 50 episódios (com exploração): {retornos[-50:].mean():.1f}")

# ======================================================================
# Bloco 3
# ======================================================================
configs = {"completo": {}, "sem rede-alvo": {"usar_alvo": False}, "sem memória": {"usar_replay": False},
           "controle: lote 1, com memória": {"lote": 1, "aquecimento": 1}}
resumo = {}
for nome, kw in configs.items():
    finais = []
    for s in range(3):
        _, r = treinar_dqn(semente=s, n_ep=200, **kw)
        finais.append(r[-50:].mean())
    resumo[nome] = finais
    print(f"{nome:14s} | retorno médio nos últimos 50 episódios, por semente: {np.round(finais, 1)}", flush=True)

# ======================================================================
# Bloco 4
# ======================================================================
for duplo in [False, True]:
    _, r = treinar_dqn(semente=3, n_ep=200, duplo=duplo)
    print(f"Double DQN = {duplo!s:5} | retorno médio dos últimos 50 episódios = {r[-50:].mean():.1f}")

# ======================================================================
# Bloco 5
# ======================================================================
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("CartPole-v1")
modelo = DQN("MlpPolicy", env, learning_rate=5e-4, buffer_size=50_000, learning_starts=1000, batch_size=64,
             gamma=0.99, target_update_interval=250, exploration_fraction=0.3, exploration_final_eps=0.05,
             policy_kwargs=dict(net_arch=[128, 128]), verbose=0, seed=0)
modelo.learn(total_timesteps=20_000)
media, desvio = evaluate_policy(modelo, env, n_eval_episodes=20)
print(f"DQN (Stable-Baselines3): retorno médio = {media:.1f} ± {desvio:.1f}")

# ======================================================================
# Bloco 6
# ======================================================================
import pandas as pd
from gymnasium import spaces
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import ParameterGrid
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

df = fetch_ucirepo(id=352).data.original.copy()
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0) & (~df["InvoiceNo"].astype(str).str.startswith("C"))]
top = df["StockCode"].value_counts().index[0]
d = df[df["StockCode"] == top].copy(); d["dia"] = pd.to_datetime(d["InvoiceDate"]).dt.normalize()
serie = d.groupby("dia")["Quantity"].sum()
serie = serie.reindex(pd.date_range(serie.index.min(), serie.index.max(), freq="D"), fill_value=0)
demanda = serie.to_numpy().astype(float); dias_sem = serie.index.dayofweek.to_numpy()
corte = int(0.7 * len(demanda)); preco_medio = float(d["UnitPrice"].median())

class EstoqueEnv(gym.Env):
    PEDIDOS = np.array([0, 10, 20, 40, 80]); EST_MAX = 200
    def __init__(self, demanda, dias_sem, horizonte=30, preco=preco_medio):
        super().__init__()
        self.dem, self.dsem, self.H, self.p = demanda, dias_sem, horizonte, preco
        self.custo, self.estoc, self.multa = 0.6 * preco, 0.01 * preco, 0.3 * preco
        self.escala = max(demanda.mean(), 1.0)
        self.action_space = spaces.Discrete(len(self.PEDIDOS))
        self.observation_space = spaces.Box(0.0, 10.0, shape=(4,), dtype=np.float32)
    def _obs(self):
        ontem = self.dem[self.i - 1] if self.i > 0 else 0.0
        media7 = self.dem[max(0, self.i - 7):self.i].mean() if self.i > 0 else 0.0
        return np.array([self.estoque / self.EST_MAX * 10, ontem / self.escala, media7 / self.escala, self.dsem[self.i] / 6 * 10], dtype=np.float32)
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.i0 = int(self.np_random.integers(7, len(self.dem) - self.H - 1)); self.i = self.i0; self.t = 0; self.estoque = 20.0
        return self._obs(), {}
    def step(self, acao):
        self.estoque = min(self.estoque + self.PEDIDOS[acao], self.EST_MAX)
        d = self.dem[self.i]; vendas = min(d, self.estoque)
        r = vendas * self.p - self.PEDIDOS[acao] * self.custo - (self.estoque - vendas) * self.estoc - (d - vendas) * self.multa
        self.estoque -= vendas; self.i += 1; self.t += 1
        return self._obs(), float(r), False, self.t >= self.H, {}

env_tr = EstoqueEnv(demanda[:corte], dias_sem[:corte]); env_te = EstoqueEnv(demanda[corte:], dias_sem[corte:])
modelo = DQN("MlpPolicy", env_tr, learning_rate=5e-4, buffer_size=30_000, learning_starts=500, batch_size=64,
             target_update_interval=250, exploration_fraction=0.3, policy_kwargs=dict(net_arch=[64, 64]), verbose=0, seed=0)
modelo.learn(total_timesteps=15_000)
print("DQN no teste:", round(evaluate_policy(modelo, env_te, n_eval_episodes=100)[0], 1))

def politica_sS(env, s, S):
    obs, _ = env.reset(seed=0); total, ret = 0.0, []
    for _ in range(100):
        obs, _ = env.reset(); fim, total = False, 0.0
        while not fim:
            falta = S - env.estoque if env.estoque < s else 0
            a = int(np.argmin(np.abs(env.PEDIDOS - falta)))               # pedido mais próximo da quantidade desejada
            obs, r, term, trunc, _ = env.step(a); total += r; fim = term or trunc
        ret.append(total)
    return np.mean(ret)

melhor = max(ParameterGrid({"s": [5, 10, 20, 40], "S": [30, 60, 100]}), key=lambda p: politica_sS(env_tr, **p))
print("melhor (s, S) no treino:", melhor, "| desempenho no teste:", round(politica_sS(env_te, **melhor), 1))

