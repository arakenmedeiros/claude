"""
Livro: Aprendizado por Reforço — Apêndice H — Introdução ao Stable-Baselines3
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
from stable_baselines3 import DQN, PPO, SAC, A2C, TD3
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("CartPole-v1")
modelo = PPO("MlpPolicy", env, verbose=0, seed=0)       # 1) algoritmo + política (rede MLP)
modelo.learn(total_timesteps=20_000)                    # 2) treinar
media, desvio = evaluate_policy(modelo, env, n_eval_episodes=10)   # 3) avaliar
print(f"PPO: retorno médio = {media:.1f} ± {desvio:.1f}")

# ======================================================================
# Bloco 2
# ======================================================================
modelo.save("ppo_cartpole")                             # salva em arquivo
carregado = PPO.load("ppo_cartpole")
obs, _ = env.reset(seed=0); total, fim = 0.0, False
while not fim:
    acao, _ = carregado.predict(obs, deterministic=True)     # ação do agente treinado
    obs, r, term, trunc, _ = env.step(int(acao)); total += r; fim = term or trunc
print("retorno de um episódio com o modelo carregado:", total)

# ======================================================================
# Bloco 3
# ======================================================================
modelo_dqn = DQN("MlpPolicy", env, learning_rate=5e-4, buffer_size=50_000, learning_starts=1000,
                 batch_size=64, gamma=0.99, target_update_interval=250, exploration_fraction=0.3,
                 exploration_final_eps=0.05, policy_kwargs=dict(net_arch=[128, 128]), verbose=0, seed=0)
modelo_dqn.learn(total_timesteps=15_000)
print("DQN (SB3):", evaluate_policy(modelo_dqn, env, n_eval_episodes=10)[0])

# ======================================================================
# Bloco 4
# ======================================================================
from stable_baselines3.common.env_util import make_vec_env
envs = make_vec_env("CartPole-v1", n_envs=4, seed=0)
modelo_a2c = A2C("MlpPolicy", envs, verbose=0, seed=0).learn(total_timesteps=10_000)
print("A2C:", evaluate_policy(modelo_a2c, gym.make("CartPole-v1"), n_eval_episodes=5)[0])

# ======================================================================
# Bloco 5
# ======================================================================
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

env_aval = Monitor(gym.make("CartPole-v1"))
cb = EvalCallback(env_aval, eval_freq=2_000, n_eval_episodes=5, deterministic=True, verbose=0)
PPO("MlpPolicy", gym.make("CartPole-v1"), verbose=0, seed=1).learn(total_timesteps=10_000, callback=cb)
print("melhor retorno médio nas avaliações:", round(float(cb.best_mean_reward), 1), "| última:", round(float(cb.last_mean_reward), 1))

