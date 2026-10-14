# Apêndice H — Introdução ao Stable-Baselines3
O **Stable-Baselines3** (SB3) é uma biblioteca com implementações confiáveis e testadas dos principais algoritmos de AR profundo, escritas em PyTorch (RAFFIN et al., 2021). Ela é ideal para **aplicar** algoritmos a ambientes seus, sem reescrever tudo.

## H.1 O primeiro treino em quatro linhas

```python
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN, PPO, SAC, A2C, TD3
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("CartPole-v1")
modelo = PPO("MlpPolicy", env, verbose=0, seed=0)       # 1) algoritmo + política (rede MLP)
modelo.learn(total_timesteps=20_000)                    # 2) treinar
media, desvio = evaluate_policy(modelo, env, n_eval_episodes=10)   # 3) avaliar
print(f"PPO: retorno médio = {media:.1f} ± {desvio:.1f}")
```

O **PPO** costuma acertar o CartPole em poucos minutos de treino.

## H.2 Escolhendo o algoritmo

| Algoritmo | Ações | Tipo | Quando usar |
|---|---|---|---|
| **DQN** | Discretas | Valor, off-policy | Jogos e controle discreto |
| **A2C / PPO** | Discretas ou contínuas | Política, on-policy | Primeira escolha geral (PPO) |
| **DDPG / TD3 / SAC** | Contínuas | Ator-crítico, off-policy | Robótica, controle contínuo (SAC e TD3 são os mais usados) |

## H.3 Salvar, carregar e usar o modelo

```python
modelo.save("ppo_cartpole")                             # salva em arquivo
carregado = PPO.load("ppo_cartpole")
obs, _ = env.reset(seed=0); total, fim = 0.0, False
while not fim:
    acao, _ = carregado.predict(obs, deterministic=True)     # ação do agente treinado
    obs, r, term, trunc, _ = env.step(int(acao)); total += r; fim = term or trunc
print("retorno de um episódio com o modelo carregado:", total)
```

## H.4 Hiperparâmetros importantes

Todo algoritmo recebe argumentos como `learning_rate`, `gamma`, `batch_size`, `n_steps` (PPO) ou `buffer_size` (algoritmos off-policy). Rede e arquitetura vão em `policy_kwargs`:

```python
modelo_dqn = DQN("MlpPolicy", env, learning_rate=5e-4, buffer_size=50_000, learning_starts=1000,
                 batch_size=64, gamma=0.99, target_update_interval=250, exploration_fraction=0.3,
                 exploration_final_eps=0.05, policy_kwargs=dict(net_arch=[128, 128]), verbose=0, seed=0)
modelo_dqn.learn(total_timesteps=15_000)
print("DQN (SB3):", evaluate_policy(modelo_dqn, env, n_eval_episodes=10)[0])
```

## H.5 Ambientes personalizados e vetorizados

O SB3 aceita qualquer ambiente que siga a interface do Gymnasium (Apêndice F). Para treinar mais rápido, use vários ambientes em paralelo:

```python
from stable_baselines3.common.env_util import make_vec_env
envs = make_vec_env("CartPole-v1", n_envs=4, seed=0)
modelo_a2c = A2C("MlpPolicy", envs, verbose=0, seed=0).learn(total_timesteps=10_000)
print("A2C:", evaluate_policy(modelo_a2c, gym.make("CartPole-v1"), n_eval_episodes=5)[0])
```

## H.6 Curvas de aprendizado e callbacks

Para acompanhar o treino, use `Monitor` e *callbacks* (funções chamadas periodicamente), como o `EvalCallback`, que avalia e guarda o melhor modelo:

```python
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

env_aval = Monitor(gym.make("CartPole-v1"))
cb = EvalCallback(env_aval, eval_freq=2_000, n_eval_episodes=5, deterministic=True, verbose=0)
PPO("MlpPolicy", gym.make("CartPole-v1"), verbose=0, seed=1).learn(total_timesteps=10_000, callback=cb)
print("avaliações:", len(cb.evaluations_results), "| melhor retorno médio:", round(float(np.max(cb.evaluations_results.mean(axis=1))), 1) if hasattr(cb.evaluations_results, "mean") else "n/d")
```

:::atencao Boas práticas
- **Várias sementes:** repita cada experimento com ao menos 3 a 5 sementes.
- **Avalie em ambiente separado**, sem exploração (`deterministic=True`).
- **Não compare algoritmos com hiperparâmetros do padrão sem pensar**: ajustar o essencial (taxa de aprendizado, tamanho de lote) importa mais do que parece.
:::
