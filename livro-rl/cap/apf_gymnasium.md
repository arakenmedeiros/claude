# Apêndice F — Introdução ao Gymnasium
O **Gymnasium** é a biblioteca padrão de ambientes de AR em Python. Ela é a sucessora do OpenAI Gym (BROCKMAN et al., 2016), hoje mantida pela Fundação Farama. Ela define uma **interface única**: qualquer algoritmo funciona com qualquer ambiente.

## F.1 O ciclo reset → step

```python
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
```

| Retorno de `step` | Significado |
|---|---|
| `obs` | Nova observação (estado) |
| `recompensa` | Recompensa do passo |
| `terminou` | O episódio acabou por uma condição **do problema** (queda, vitória, derrota) |
| `truncou` | O episódio foi **cortado** por limite de tempo (ex.: 500 passos) |
| `info` | Dados extras (para depuração) |

:::atencao terminou × truncou
Na hora de calcular o alvo TD, use apenas `terminou` para zerar o valor futuro. Se o episódio foi **truncado** por limite de tempo, o estado seguinte ainda tem valor. Misturar os dois é um erro comum.
:::

## F.2 Espaços de ações e observações

```python
print(env.action_space)                   # Discrete(2): ações 0 e 1
print(env.observation_space)              # Box: 4 números reais, com limites
print(env.observation_space.shape, env.action_space.n)

amb = gym.make("Pendulum-v1")
print(amb.action_space)                   # Box(-2, 2, (1,)): ação contínua
```

- **Discrete(n):** n ações inteiras (0 a n−1).
- **Box:** vetores de números reais (contínuos).
- **MultiDiscrete**, **Dict** e **Tuple** combinam espaços.

## F.3 Ambientes usados neste livro

| Ambiente | Tipo | Capítulo |
|---|---|---|
| `CartPole-v1` | Estado contínuo, 2 ações | 1, 4, 6–10 |
| `FrozenLake-v1` | Tabular, estocástico | 3, 4 |
| `Blackjack-v1` | Tabular, episódico | 5 |
| `CliffWalking-v1` | Tabular | 6 |
| `Taxi-v4` (ou `v3`) | Tabular, 500 estados | 6 |
| `MountainCar-v0` | Estado contínuo, 3 ações | 7 |
| `Pendulum-v1` | Ação contínua | 10 |
| Ambientes `box2d` (ex.: LunarLander, BipedalWalker) | Requerem `pip install "gymnasium[box2d]"` | 10 (opcional) |

Para ver todos os registrados: `list(gym.registry.keys())`. Alguns exigem pacotes extras: `pip install "gymnasium[classic-control,toy-text,box2d,mujoco]"`.

## F.4 Criando seu próprio ambiente

Para um problema seu, crie uma **classe** que herda de `gym.Env` e defina `reset` e `step`. Exemplo: um **cassino com 3 máquinas** (bandit) como ambiente.

```python
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
```

:::dica Teste seu ambiente
O Gymnasium tem um verificador: `from gymnasium.utils.env_checker import check_env; check_env(meu_ambiente)`. Ele aponta erros comuns antes de você treinar por horas.
:::

## F.5 Wrappers

Um *wrapper* modifica um ambiente sem alterar seu código: limitar episódios, normalizar observações, empilhar quadros, registrar vídeos.

```python
from gymnasium.wrappers import TimeLimit, RecordEpisodeStatistics

env = RecordEpisodeStatistics(TimeLimit(gym.make("CartPole-v1"), max_episode_steps=100))
obs, _ = env.reset(seed=0); fim = False
while not fim:
    obs, r, term, trunc, info = env.step(env.action_space.sample()); fim = term or trunc
print("estatísticas do episódio:", {k: float(v) for k, v in info["episode"].items() if k in ("r", "l")})
```

## F.6 Renderização

Passe `render_mode="human"` para abrir uma janela (precisa de `pygame`) ou `"rgb_array"` para obter imagens como arrays. Em servidores e no Colab, use `rgb_array` e salve quadros ou vídeo.

## F.7 Ambientes vetorizados

Para acelerar o treino, rode vários ambientes em paralelo:

```python
envs = gym.make_vec("CartPole-v1", num_envs=4, vectorization_mode="sync")
obs, _ = envs.reset(seed=0)
obs, r, term, trunc, _ = envs.step(np.array([0, 1, 0, 1]))
print(obs.shape, r)
envs.close()
```
