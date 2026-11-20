"""
Livro: Aprendizado por Reforço — Apêndice F — Introdução ao Gymnasium
Autor do livro: Araken de Medeiros Santos

Os blocos abaixo seguem a ordem do texto e devem ser executados em sequência (como células de um notebook).
Os trechos marcados como 'não executado' (por exemplo, downloads do Kaggle) aparecem comentados.
"""
import warnings; warnings.filterwarnings('ignore')

# ======================================================================
# Bloco 1
# ======================================================================
import gymnasium as gym
import numpy as np

env = gym.make("CartPole-v1")
obs, info = env.reset(seed=0)                 # começa um episódio
print("observação:", obs.round(3))

total, fim = 0.0, False
while not fim:
    acao = env.action_space.sample()          # ação aleatória
    obs, recompensa, terminou, truncou, info = env.step(acao)
    total += recompensa
    fim = terminou or truncou
print("retorno do episódio:", total)
env.close()

# ======================================================================
# Bloco 2
# ======================================================================
print(env.action_space)                   # Discrete(2): ações 0 e 1
print(env.observation_space)              # Box: 4 números reais, com limites
print(env.observation_space.shape, env.action_space.n)

amb = gym.make("Pendulum-v1")
print(amb.action_space)                   # Box(-2, 2, (1,)): ação contínua

# ======================================================================
# Bloco 3
# ======================================================================
from gymnasium import spaces

class BanditEnv(gym.Env):
    def __init__(self, p=(0.2, 0.5, 0.7), horizonte=20):
        super().__init__()
        self.p, self.horizonte = np.array(p), horizonte
        self.action_space = spaces.Discrete(len(p))
        self.observation_space = spaces.Discrete(1)          # só existe uma situação
    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self.t = 0
        return 0, {}
    def step(self, acao):
        recompensa = float(self.np_random.random() < self.p[acao])
        self.t += 1
        return 0, recompensa, False, self.t >= self.horizonte, {}

env = BanditEnv()
obs, _ = env.reset(seed=1); total, fim = 0.0, False
while not fim:
    obs, r, term, trunc, _ = env.step(2); total += r; fim = term or trunc   # sempre o melhor braço
print("retorno de 20 jogadas no braço 2:", total)

# ======================================================================
# Bloco 4
# ======================================================================
from gymnasium.wrappers import TimeLimit, RecordEpisodeStatistics

env = RecordEpisodeStatistics(TimeLimit(gym.make("CartPole-v1"), max_episode_steps=100))
obs, _ = env.reset(seed=0); fim = False
while not fim:
    obs, r, term, trunc, info = env.step(env.action_space.sample()); fim = term or trunc
print("estatísticas do episódio:", {k: float(v) for k, v in info["episode"].items() if k in ("r", "l")})

# ======================================================================
# Bloco 5
# ======================================================================
envs = gym.make_vec("CartPole-v1", num_envs=4, vectorization_mode="sync")
obs, _ = envs.reset(seed=0)
obs, r, term, trunc, _ = envs.step(np.array([0, 1, 0, 1]))
print(obs.shape, r)
envs.close()

