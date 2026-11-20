"""
Livro: Aprendizado por Reforço — Gradientes de Política e Atores-Críticos
Autor do livro: Araken de Medeiros Santos

Os blocos abaixo seguem a ordem do texto e devem ser executados em sequência (como células de um notebook).
Os trechos marcados como 'não executado' (por exemplo, downloads do Kaggle) aparecem comentados.
"""
import warnings; warnings.filterwarnings('ignore')

# ======================================================================
# Bloco 1
# ======================================================================
import numpy as np

rng = np.random.default_rng(0)
p = np.array([0.3, 0.4, 0.3]); mu = np.array([9.0, 10.0, 11.0]); N = 20_000

def estimativas_gradiente(base):
    g = np.zeros((N, 3))
    for i in range(N):
        a = rng.choice(3, p=p); G = rng.normal(mu[a], 1.0)
        e = -p.copy(); e[a] += 1.0                       # ∇ ln π em relação às preferências (softmax)
        g[i] = e * (G - base)
    return g

g0, g1 = estimativas_gradiente(0.0), estimativas_gradiente(10.0)
print("gradiente médio sem linha de base :", g0.mean(0).round(2))
print("gradiente médio com linha de base :", g1.mean(0).round(2))
print("desvio-padrão sem linha de base   :", g0.std(0).round(2))
print("desvio-padrão com linha de base   :", g1.std(0).round(2))

# ======================================================================
# Bloco 2
# ======================================================================
import random, time
import torch, torch.nn as nn, gymnasium as gym
from torch.distributions import Categorical
torch.set_num_threads(1)

def rede(n_in, n_out, ocultas=64):
    return nn.Sequential(nn.Linear(n_in, ocultas), nn.Tanh(), nn.Linear(ocultas, ocultas), nn.Tanh(), nn.Linear(ocultas, n_out))

def retornos_descontados(recompensas, gamma):
    G, out = 0.0, []
    for r in reversed(recompensas):
        G = r + gamma * G; out.append(G)
    return out[::-1]

def treinar_reinforce(semente=0, n_ep=600, baseline=False, lr=3e-3, gamma=0.99, entropia=0.0, verbose=False):
    random.seed(semente); np.random.seed(semente); torch.manual_seed(semente)
    env = gym.make("CartPole-v1"); n_obs, n_acoes = env.observation_space.shape[0], env.action_space.n
    politica = rede(n_obs, n_acoes); valor = rede(n_obs, 1)
    opt_p = torch.optim.Adam(politica.parameters(), lr=lr); opt_v = torch.optim.Adam(valor.parameters(), lr=1e-2)
    historico = []
    for ep in range(n_ep):
        obs, _ = env.reset(seed=semente * 10_000 + ep); fim = False; estados, logps, ents, recs = [], [], [], []
        while not fim:
            s = torch.as_tensor(obs, dtype=torch.float32); dist = Categorical(logits=politica(s)); a = dist.sample()
            obs, r, term, trunc, _ = env.step(int(a)); fim = term or trunc
            estados.append(s); logps.append(dist.log_prob(a)); ents.append(dist.entropy()); recs.append(r)
        G = torch.as_tensor(retornos_descontados(recs, gamma), dtype=torch.float32)
        logps, ents = torch.stack(logps), torch.stack(ents)
        if baseline:
            v = valor(torch.stack(estados)).squeeze(1)
            vantagem = (G - v).detach()                                # G − b(s)
            perda_v = nn.functional.mse_loss(v, G); opt_v.zero_grad(); perda_v.backward(); opt_v.step()
        else:
            vantagem = G
        perda = -(logps * vantagem).mean() - entropia * ents.mean()    # ascensão do gradiente
        opt_p.zero_grad(); perda.backward(); opt_p.step()
        historico.append(sum(recs))
        if verbose and (ep + 1) % 100 == 0:
            print(f"episódio {ep+1:4d} | retorno médio (últimos 100) = {np.mean(historico[-100:]):6.1f}", flush=True)
    return politica, np.array(historico)

def treinar_ator_critico(semente=0, n_ep=600, n_passos=20, lr_p=1e-3, lr_v=2e-3, gamma=0.99, entropia=0.01, verbose=False):
    random.seed(semente); np.random.seed(semente); torch.manual_seed(semente)
    env = gym.make("CartPole-v1"); n_obs, n_acoes = env.observation_space.shape[0], env.action_space.n
    ator = rede(n_obs, n_acoes); critico = rede(n_obs, 1)
    opt_a = torch.optim.Adam(ator.parameters(), lr=lr_p); opt_c = torch.optim.Adam(critico.parameters(), lr=lr_v)
    historico = []
    for ep in range(n_ep):
        obs, _ = env.reset(seed=semente * 10_000 + ep); fim, total = False, 0.0
        buf_s, buf_logp, buf_ent, buf_r, buf_term = [], [], [], [], []
        while not fim:
            s = torch.as_tensor(obs, dtype=torch.float32); dist = Categorical(logits=ator(s)); a = dist.sample()
            obs, r, term, trunc, _ = env.step(int(a)); fim = term or trunc; total += r
            buf_s.append(s); buf_logp.append(dist.log_prob(a)); buf_ent.append(dist.entropy()); buf_r.append(r); buf_term.append(term)
            if len(buf_r) == n_passos or fim:
                with torch.no_grad(): R = 0.0 if buf_term[-1] else critico(torch.as_tensor(obs, dtype=torch.float32)).item()
                alvos = []
                for rr in reversed(buf_r): R = rr + gamma * R; alvos.append(R)
                alvos = torch.as_tensor(alvos[::-1], dtype=torch.float32)
                v = critico(torch.stack(buf_s)).squeeze(1); vantagem = (alvos - v).detach()
                perda_c = nn.functional.mse_loss(v, alvos); opt_c.zero_grad(); perda_c.backward(); opt_c.step()
                perda_a = -(torch.stack(buf_logp) * vantagem).mean() - entropia * torch.stack(buf_ent).mean()
                opt_a.zero_grad(); perda_a.backward(); opt_a.step()
                buf_s, buf_logp, buf_ent, buf_r, buf_term = [], [], [], [], []
        historico.append(total)
        if verbose and (ep + 1) % 100 == 0: print(f"episódio {ep+1:4d} | retorno médio (últimos 100) = {np.mean(historico[-100:]):6.1f}", flush=True)
    return ator, np.array(historico)

for nome, f, kw in [("REINFORCE", treinar_reinforce, {}), ("REINFORCE + linha de base", treinar_reinforce, {"baseline": True}),
                    ("Ator-crítico", treinar_ator_critico, {})]:
    t0 = time.time(); _, h = f(semente=0, n_ep=600, **kw)
    print(f"{nome:26s} | retorno médio: episódios 1–100 = {h[:100].mean():6.1f} | 501–600 = {h[-100:].mean():6.1f} | máx = {h.max():.0f} | {time.time()-t0:.0f} s", flush=True)

# ======================================================================
# Bloco 3
# ======================================================================
from sklearn.utils import resample

politica_final, _ = treinar_reinforce(semente=1, n_ep=500, baseline=True)
def avaliar_politica(pol, n=30):
    env = gym.make("CartPole-v1"); ret = []
    for i in range(n):
        obs, _ = env.reset(seed=90_000 + i); fim, total = False, 0.0
        while not fim:
            with torch.no_grad(): a = int(pol(torch.as_tensor(obs, dtype=torch.float32)).argmax())
            obs, r, term, trunc, _ = env.step(a); total += r; fim = term or trunc
        ret.append(total)
    return np.array(ret)

ret = avaliar_politica(politica_final)
boot = [resample(ret, random_state=b).mean() for b in range(300)]
print(f"Política gulosa: retorno médio = {ret.mean():.1f} (IC 95%: {np.percentile(boot, 2.5):.1f} a {np.percentile(boot, 97.5):.1f})")

# ======================================================================
# Bloco 4
# ======================================================================
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

banco = fetch_ucirepo(id=222)
X = banco.data.features.drop(columns=["duration"], errors="ignore").copy()
y = (banco.data.targets.iloc[:, 0] == "yes").astype(int).to_numpy()
cat = X.select_dtypes(exclude="number").columns.tolist(); num = X.select_dtypes(include="number").columns.tolist()
X[cat] = X[cat].fillna("desconhecido")
prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat), ("num", StandardScaler(), num)])
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
Z_tr = prep.fit_transform(X_tr).astype(np.float32); Z_te = prep.transform(X_te).astype(np.float32)

VALOR, CUSTO = 100.0, 8.0
rec_ligar = lambda yy: VALOR * yy - CUSTO
print(f"ligar para todos: {rec_ligar(y_te).mean():.2f} | não ligar: 0.00 | oráculo: {np.maximum(rec_ligar(y_te), 0).mean():.2f}")
clf = LogisticRegression(max_iter=1000).fit(Z_tr, y_tr)
liga = clf.predict_proba(Z_te)[:, 1] > CUSTO / VALOR
print(f"classificador com limiar {CUSTO/VALOR:.2f}: {(liga * rec_ligar(y_te)).mean():.2f}")

torch.manual_seed(0); rng_l = np.random.default_rng(0)
pol = nn.Sequential(nn.Linear(Z_tr.shape[1], 32), nn.ReLU(), nn.Linear(32, 2))      # ações: 0 = não ligar, 1 = ligar
opt = torch.optim.Adam(pol.parameters(), lr=3e-3); Zt = torch.as_tensor(Z_tr); media_rec = 0.0
for it in range(600):
    idx = rng_l.integers(0, len(Zt), 256); dist = Categorical(logits=pol(Zt[idx])); a = dist.sample()
    r = torch.as_tensor(np.where(a.numpy() == 1, rec_ligar(y_tr[idx]), 0.0), dtype=torch.float32)
    media_rec = 0.95 * media_rec + 0.05 * r.mean().item()                              # linha de base: média móvel
    perda = -(dist.log_prob(a) * (r - media_rec)).mean() - 0.01 * dist.entropy().mean()
    opt.zero_grad(); perda.backward(); opt.step()
    if (it + 1) % 200 == 0:
        with torch.no_grad(): p_lig = torch.softmax(pol(torch.as_tensor(Z_te)), 1)[:, 1].numpy()
        print(f"iteração {it+1:3d} | recompensa esperada no teste = {(p_lig * rec_ligar(y_te)).mean():.2f}")

