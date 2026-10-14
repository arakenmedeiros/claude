import random, time
import numpy as np, torch, torch.nn as nn, gymnasium as gym
from torch.distributions import Categorical
torch.set_num_threads(1)

def rede(n_in, n_out, ocultas=64):
    return nn.Sequential(nn.Linear(n_in, ocultas), nn.Tanh(), nn.Linear(ocultas, ocultas), nn.Tanh(), nn.Linear(ocultas, n_out))

def retornos_descontados(recompensas, gamma):
    G, out = 0.0, []
    for r in reversed(recompensas):
        G = r + gamma * G; out.append(G)
    return out[::-1]

def treinar_reinforce(semente=0, n_ep=600, baseline=False, lr=3e-3, gamma=0.99, entropia=0.0, env_id="CartPole-v1", verbose=False):
    random.seed(semente); np.random.seed(semente); torch.manual_seed(semente)
    env = gym.make(env_id); n_obs, n_acoes = env.observation_space.shape[0], env.action_space.n
    politica = rede(n_obs, n_acoes); valor = rede(n_obs, 1)
    opt_p = torch.optim.Adam(politica.parameters(), lr=lr); opt_v = torch.optim.Adam(valor.parameters(), lr=1e-2)
    historico = []
    for ep in range(n_ep):
        obs, _ = env.reset(seed=semente * 10_000 + ep); fim = False
        estados, logps, ents, recs = [], [], [], []
        while not fim:
            s = torch.as_tensor(obs, dtype=torch.float32); dist = Categorical(logits=politica(s)); a = dist.sample()
            obs, r, term, trunc, _ = env.step(int(a)); fim = term or trunc
            estados.append(s); logps.append(dist.log_prob(a)); ents.append(dist.entropy()); recs.append(r)
        G = torch.as_tensor(retornos_descontados(recs, gamma), dtype=torch.float32)
        logps = torch.stack(logps); ents = torch.stack(ents)
        if baseline:
            S = torch.stack(estados); v = valor(S).squeeze(1)
            vantagem = (G - v).detach()                                  # G − b(s): subtrai a linha de base
            perda_v = nn.functional.mse_loss(v, G); opt_v.zero_grad(); perda_v.backward(); opt_v.step()
        else:
            vantagem = G
        perda = -(logps * vantagem).mean() - entropia * ents.mean()      # ascensão do gradiente = descida do negativo
        opt_p.zero_grad(); perda.backward(); opt_p.step()
        historico.append(sum(recs))
        if verbose and (ep + 1) % 100 == 0: print(f"episódio {ep+1:4d} | retorno médio (últimos 100) = {np.mean(historico[-100:]):6.1f}", flush=True)
    return politica, np.array(historico)

def treinar_ator_critico(semente=0, n_ep=600, lr_p=2e-3, lr_v=5e-3, gamma=0.99, entropia=0.01, env_id="CartPole-v1", verbose=False):
    random.seed(semente); np.random.seed(semente); torch.manual_seed(semente)
    env = gym.make(env_id); n_obs, n_acoes = env.observation_space.shape[0], env.action_space.n
    ator = rede(n_obs, n_acoes); critico = rede(n_obs, 1)
    opt_a = torch.optim.Adam(ator.parameters(), lr=lr_p); opt_c = torch.optim.Adam(critico.parameters(), lr=lr_v)
    historico = []
    for ep in range(n_ep):
        obs, _ = env.reset(seed=semente * 10_000 + ep); fim, total = False, 0.0
        while not fim:
            s = torch.as_tensor(obs, dtype=torch.float32); dist = Categorical(logits=ator(s)); a = dist.sample()
            obs2, r, term, trunc, _ = env.step(int(a)); fim = term or trunc; total += r
            s2 = torch.as_tensor(obs2, dtype=torch.float32)
            with torch.no_grad(): alvo = r + gamma * (0.0 if term else critico(s2).item())          # alvo TD
            v = critico(s).squeeze()
            delta = alvo - v                                                                         # erro TD = vantagem
            perda_c = delta ** 2; opt_c.zero_grad(); perda_c.backward(); opt_c.step()
            perda_a = -dist.log_prob(a) * delta.detach() - entropia * dist.entropy()
            opt_a.zero_grad(); perda_a.backward(); opt_a.step(); obs = obs2
        historico.append(total)
        if verbose and (ep + 1) % 100 == 0: print(f"episódio {ep+1:4d} | retorno médio (últimos 100) = {np.mean(historico[-100:]):6.1f}", flush=True)
    return ator, np.array(historico)
