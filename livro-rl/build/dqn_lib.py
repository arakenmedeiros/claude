import random, time
from collections import deque
import numpy as np, torch, torch.nn as nn, gymnasium as gym
torch.set_num_threads(1)

def criar_rede(n_obs, n_acoes, ocultas=128):
    return nn.Sequential(nn.Linear(n_obs, ocultas), nn.ReLU(), nn.Linear(ocultas, ocultas), nn.ReLU(), nn.Linear(ocultas, n_acoes))

def treinar_dqn(semente=0, n_ep=300, usar_replay=True, usar_alvo=True, duplo=False, lr=5e-4, gamma=0.99,
                lote=64, tam_memoria=50_000, aquecimento=1000, troca_alvo=250, decaimento_eps=5000, env_id="CartPole-v1", verbose=False):
    random.seed(semente); np.random.seed(semente); torch.manual_seed(semente)
    env = gym.make(env_id); n_obs = env.observation_space.shape[0]; n_acoes = env.action_space.n
    q = criar_rede(n_obs, n_acoes); q_alvo = criar_rede(n_obs, n_acoes); q_alvo.load_state_dict(q.state_dict())
    if not usar_alvo: q_alvo = q                                  # sem rede-alvo: o alvo usa a própria rede
    if not usar_replay: tam_memoria, lote, aquecimento = 1, 1, 1  # sem memória: aprende só com a transição mais recente
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    memoria = deque(maxlen=tam_memoria); passo = 0; retornos = []
    for ep in range(n_ep):
        obs, _ = env.reset(seed=semente * 10_000 + ep); fim, total = False, 0.0
        while not fim:
            eps = max(0.05, 1 - passo / decaimento_eps)
            if random.random() < eps: a = env.action_space.sample()
            else:
                with torch.no_grad(): a = int(q(torch.as_tensor(obs, dtype=torch.float32)).argmax())
            obs2, r, term, trunc, _ = env.step(a); fim = term or trunc; total += r
            memoria.append((obs, a, r, obs2, float(term))); obs = obs2; passo += 1
            if len(memoria) >= aquecimento:
                O, A, R, O2, T = map(np.array, zip(*random.sample(memoria, min(lote, len(memoria)))))
                O, O2, R, T = [torch.as_tensor(x, dtype=torch.float32) for x in (O, O2, R, T)]; A = torch.as_tensor(A)
                with torch.no_grad():
                    if duplo:                                        # Double DQN: a rede online escolhe, a rede-alvo avalia
                        a_star = q(O2).argmax(1, keepdim=True); proximo = q_alvo(O2).gather(1, a_star).squeeze(1)
                    else:
                        proximo = q_alvo(O2).max(1).values
                    y = R + gamma * (1 - T) * proximo
                pred = q(O).gather(1, A.unsqueeze(1)).squeeze(1)
                perda = nn.functional.smooth_l1_loss(pred, y)        # perda de Huber
                opt.zero_grad(); perda.backward(); nn.utils.clip_grad_norm_(q.parameters(), 10.0); opt.step()
            if usar_alvo and passo % troca_alvo == 0: q_alvo.load_state_dict(q.state_dict())
        retornos.append(total)
        if verbose and (ep + 1) % 50 == 0: print(f"episódio {ep+1:4d} | retorno médio (últimos 50) = {np.mean(retornos[-50:]):6.1f}", flush=True)
    return q, np.array(retornos)
