"""Capítulo 7: soluções dos exercícios práticos e do projeto. Cada "# %%" é uma célula independente."""

# %% [Exercício 7.1: Modelo de recompensa: quantos pares são necessários?]
import numpy as np
from sklearn.linear_model import LogisticRegression

sig = lambda x: 1 / (1 + np.exp(-x))
w_ver = np.array([2.0, 1.0, -0.5])

def gerar_pares(n, rng):
    feats = lambda m: np.column_stack([rng.random(m) < 0.6, rng.random(m) < 0.4, rng.random(m)]).astype(float)
    A, B = feats(n), feats(n)
    a_vence = rng.random(n) < sig(A @ w_ver - B @ w_ver)
    return np.where(a_vence[:, None], A, B), np.where(a_vence[:, None], B, A)

rng = np.random.default_rng(0)
Wt, Lt = gerar_pares(5000, rng)
print(f"{'pares':>6} {'coeficientes':>26} {'erro ||w-ŵ||':>13} {'acurácia':>9}")
for n in (100, 500, 2000, 8000):
    W, L = gerar_pares(n, rng)
    D = W - L
    X = np.vstack([D, -D]); y = np.r_[np.ones(n), np.zeros(n)]
    rm = LogisticRegression(fit_intercept=False, C=100).fit(X, y)
    c = rm.coef_[0]
    print(f"{n:>6} {str(c.round(2)):>26} {np.linalg.norm(c - w_ver):>13.3f} {(rm.decision_function(Wt - Lt) > 0).mean():>9.3f}")


# %% [Exercício 7.2: DPO tabular e o efeito de $\beta$]
import numpy as np

sig = lambda x: 1 / (1 + np.exp(-x))
def log_softmax(z):
    z = z - z.max()
    return z - np.log(np.exp(z).sum())

pref = np.array([0.50, 0.30, 0.15, 0.05]); lref = np.log(pref)
r = np.array([0.0, 1.0, 2.0, 3.0]); I = np.eye(4)
rng = np.random.default_rng(0)
N = 4000
a, b = rng.integers(0, 4, N), rng.integers(0, 4, N)
m = a != b
a, b = a[m], b[m]
vence_a = rng.random(len(a)) < sig(r[a] - r[b])
W, L = np.where(vence_a, a, b), np.where(vence_a, b, a)

print(f"{'beta':>5} | {'política DPO':<26} | {'política ótima (teoria)':<26} | {'dif. máx.':>9} | {'recompensa DPO':>14}")
for beta in (0.25, 0.5, 1.0, 2.0):
    theta = lref.copy()
    for _ in range(3000):
        lp = log_softmax(theta)
        h = beta * ((lp[W] - lref[W]) - (lp[L] - lref[L]))
        grad = (-(1 - sig(h))[:, None] * beta * (I[W] - I[L])).mean(axis=0)
        theta -= (1.0 / beta) * grad
    pol = np.exp(log_softmax(theta))
    alvo = pref * np.exp(r / beta); alvo /= alvo.sum()
    print(f"{beta:>5} | {str(pol.round(3)):<26} | {str(alvo.round(3)):<26} | {np.abs(pol - alvo).max():>9.3f} | {float(pol @ r):>14.3f}")


# %% [Exercício 7.3: Selecionar a melhor entre $n$ respostas com um modelo de recompensa viesado]
import numpy as np
from sklearn.linear_model import LogisticRegression

sig = lambda x: 1 / (1 + np.exp(-x))
rng = np.random.default_rng(0)
feats = lambda *s: np.stack([rng.random(s) < 0.6, rng.random(s) < 0.4, rng.random(s)], axis=-1).astype(float)

w_obs = np.array([2.0, 1.0, 0.8])        # recompensa que os avaliadores usam (preferem respostas longas)
q_ver = np.array([2.0, 1.0, -0.5])       # qualidade verdadeira que queremos maximizar

A, B = feats(6000), feats(6000)
a_vence = rng.random(6000) < sig(A @ w_obs - B @ w_obs)
W, L = np.where(a_vence[:, None], A, B), np.where(a_vence[:, None], B, A)
D = W - L
rm = LogisticRegression(fit_intercept=False, C=100).fit(np.vstack([D, -D]), np.r_[np.ones(len(D)), np.zeros(len(D))])
print("coeficientes do modelo de recompensa (correta, cita, comprimento):", rm.coef_[0].round(2))

print(f"\n{'n':>3} {'qualidade verdadeira média':>27} {'comprimento médio':>18} {'% corretas':>11}")
for n in (1, 2, 4, 8, 16, 32):
    C = feats(5000, n)                                  # (pedidos, candidatas, 3)
    escolha = (C @ rm.coef_[0]).argmax(axis=1)
    esc = C[np.arange(5000), escolha]
    print(f"{n:>3} {float((esc @ q_ver).mean()):>27.3f} {float(esc[:, 2].mean()):>18.3f} {float(esc[:, 0].mean()):>11.1%}")


# %% [Projeto: Auditoria de um modelo de recompensa]
import numpy as np
from sklearn.linear_model import LogisticRegression

sig = lambda x: 1 / (1 + np.exp(-x))
rng = np.random.default_rng(0)

def feats(*s):
    return np.stack([rng.random(s) < 0.6, rng.random(s) < 0.4, rng.random(s), rng.random(s) < 0.5], axis=-1).astype(float)

w_aval = np.array([2.0, 1.0, 0.8, 1.2])      # correta, cita, comprimento, confiante (usada pelos avaliadores)
q_ver = np.array([2.0, 1.0, 0.0, 0.0])       # qualidade verdadeira: só correção e fonte

N = 8000
A, B = feats(N), feats(N)
a_vence = rng.random(N) < sig(A @ w_aval - B @ w_aval)
W, L = np.where(a_vence[:, None], A, B), np.where(a_vence[:, None], B, A)
D = W - L

def treinar(cols):
    X = np.vstack([D[:, cols], -D[:, cols]]); y = np.r_[np.ones(N), np.zeros(N)]
    return LogisticRegression(fit_intercept=False, C=100).fit(X, y)

completo = treinar([0, 1, 2, 3]); limpo = treinar([0, 1])
print("coeficientes do modelo completo (correta, cita, comprimento, confiante):", completo.coef_[0].round(2))
print("coeficientes do modelo limpo (correta, cita):", limpo.coef_[0].round(2))

# teste contrafactual
base = feats(2000)
mudou_tom = base.copy(); mudou_tom[:, 3] = 1.0
base_tom0 = base.copy(); base_tom0[:, 3] = 0.0
mudou_comp = base.copy(); mudou_comp[:, 2] = 0.9
base_comp = base.copy(); base_comp[:, 2] = 0.1
w = completo.coef_[0]
print(f"\naumento médio da pontuação ao ligar o tom confiante: {float(((mudou_tom - base_tom0) @ w).mean()):+.2f}")
print(f"aumento médio da pontuação ao aumentar o comprimento de 0,1 para 0,9: {float(((mudou_comp - base_comp) @ w).mean()):+.2f}")

# seleção best-of-8
C = feats(5000, 8)
print(f"\n{'modelo':<22} {'qualidade verdadeira média':>27} {'% corretas':>11} {'comprimento médio':>18}")
for nome, modelo, cols in [("completo (viesado)", completo, [0, 1, 2, 3]), ("limpo (mitigado)", limpo, [0, 1])]:
    esc = C[np.arange(5000), (C[:, :, cols] @ modelo.coef_[0]).argmax(axis=1)]
    print(f"{nome:<22} {float((esc @ q_ver).mean()):>27.3f} {float(esc[:, 0].mean()):>11.1%} {float(esc[:, 2].mean()):>18.3f}")
print(f"{'aleatório (1 candidata)':<22} {float((C[:, 0] @ q_ver).mean()):>27.3f} {float(C[:, 0, 0].mean()):>11.1%} {float(C[:, 0, 2].mean()):>18.3f}")

