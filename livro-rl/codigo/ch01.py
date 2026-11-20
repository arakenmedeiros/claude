"""
Livro: Aprendizado por Reforço — Introdução ao Aprendizado por Reforço
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

env = gym.make("CartPole-v1")
obs, info = env.reset(seed=0)

print("Observação inicial:", np.round(obs, 3))
print("Espaço de ações:", env.action_space)
print("Espaço de observações:", env.observation_space.shape)

# ======================================================================
# Bloco 2
# ======================================================================
def jogar_episodio(env, politica, rng):
    """Joga um episódio. Devolve estados, ações e o retorno total."""
    obs, _ = env.reset(seed=int(rng.integers(1_000_000)))
    estados, acoes, retorno, fim = [], [], 0.0, False
    while not fim:
        a = politica(obs)
        estados.append(obs)
        acoes.append(a)
        obs, r, terminou, truncou, _ = env.step(a)
        retorno += r
        fim = terminou or truncou
    return np.array(estados), np.array(acoes), retorno

rng = np.random.default_rng(42)
aleatoria = lambda obs: int(rng.integers(2))

retornos = [jogar_episodio(env, aleatoria, rng)[2] for _ in range(200)]
print(f"Agente aleatório: média = {np.mean(retornos):.1f}, "
      f"mín = {min(retornos):.0f}, máx = {max(retornos):.0f}")

# ======================================================================
# Bloco 3
# ======================================================================
from sklearn.tree import DecisionTreeClassifier

def politica_da_arvore(arvore, rng):
    def pol(obs):
        p = arvore.predict_proba(obs.reshape(1, -1))[0]
        return int(rng.choice(arvore.classes_, p=p))
    return pol

arvore, medias = None, []
for it in range(1, 16):
    pol = aleatoria if arvore is None else politica_da_arvore(arvore, rng)
    episodios = [jogar_episodio(env, pol, rng) for _ in range(100)]
    ret = np.array([e[2] for e in episodios])
    limiar = np.percentile(ret, 70)                 # fronteira da elite
    elite = [e for e in episodios if e[2] >= limiar]
    X = np.vstack([e[0] for e in elite])            # estados da elite
    y = np.concatenate([e[1] for e in elite])       # ações da elite
    arvore = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X, y)
    medias.append(ret.mean())
    print(f"iteração {it:2d} | retorno médio = {ret.mean():6.1f} | máx = {ret.max():.0f}")

# ======================================================================
# Bloco 4
# ======================================================================
from sklearn.utils import resample

pol_final = politica_da_arvore(arvore, rng)
ret_final = np.array([jogar_episodio(env, pol_final, rng)[2] for _ in range(200)])

boot = [resample(ret_final, random_state=s).mean() for s in range(1000)]
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"Retorno médio = {ret_final.mean():.1f}  (IC 95%: {lo:.1f} a {hi:.1f})")
print(f"Agente aleatório = {np.mean(retornos):.1f}")

# ======================================================================
# Bloco 5
# ======================================================================
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

cogumelos = fetch_ucirepo(id=73)                      # Mushroom, UCI
X = cogumelos.data.features.fillna("ausente")
venenoso = (cogumelos.data.targets.iloc[:, 0] == "p").astype(int)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, venenoso, test_size=0.3, random_state=0, stratify=venenoso)

# Modelo propositalmente simples, para que haja incerteza
modelo = make_pipeline(OneHotEncoder(handle_unknown="ignore"),
                       DecisionTreeClassifier(max_depth=2, random_state=0))
modelo.fit(X_tr, y_tr)
p_venenoso = modelo.predict_proba(X_te)[:, 1]

def recompensa_media(limiar):
    come = p_venenoso < limiar                        # política: comer se p < limiar
    r = np.where(come, np.where(y_te == 1, -35, 5), 0)
    return r.mean()

for limiar in [0.05, 0.125, 0.3, 0.5, 0.9]:
    print(f"limiar = {limiar:5.3f} | recompensa média por cogumelo = {recompensa_media(limiar):6.2f}")

# ======================================================================
# Bloco 6
# ======================================================================
import pandas as pd

def treinar(profundidade, percentil, iteracoes, semente, n_ep=60):
    rng_l = np.random.default_rng(semente)
    arv, hist = None, []
    alea = lambda obs: int(rng_l.integers(2))
    for _ in range(iteracoes):
        pol = alea if arv is None else politica_da_arvore(arv, rng_l)
        eps = [jogar_episodio(env, pol, rng_l) for _ in range(n_ep)]
        ret = np.array([e[2] for e in eps])
        lim = np.percentile(ret, percentil)
        elite = [e for e in eps if e[2] >= lim]
        X_ = np.vstack([e[0] for e in elite]); y_ = np.concatenate([e[1] for e in elite])
        arv = DecisionTreeClassifier(max_depth=profundidade, random_state=0).fit(X_, y_)
        hist.append(ret.mean())
    return arv, hist

linhas = []
for prof in [2, 4]:
    for perc in [50, 80]:
        arv, hist = treinar(prof, perc, iteracoes=8, semente=1)
        pol = politica_da_arvore(arv, rng)
        r = np.array([jogar_episodio(env, pol, rng)[2] for _ in range(30)])
        linhas.append({"profundidade": prof, "percentil": perc, "retorno_medio": r.mean()})
print(pd.DataFrame(linhas))

