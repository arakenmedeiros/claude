"""Capítulo 7: blocos de código do texto, em ordem. Cada "# %%" é uma célula."""

# %% [bloco 1]
import numpy as np

tokens = ["<sistema>", "Você", "é", "assistente", "<usuário>", "prazo", "?", "<assistente>", "Até", "a", "8ª", "semana", "<fim>"]
papel  = ["instr", "instr", "instr", "instr", "instr", "instr", "instr", "instr", "resp", "resp", "resp", "resp", "resp"]
mascara = np.array([1.0 if p == "resp" else 0.0 for p in papel])

# perdas por token ilustrativas (nats): alta nas instruções, menor nas respostas
perdas = np.array([2.1, 1.8, 1.5, 2.2, 1.2, 2.5, 1.1, 0.8, 1.4, 0.7, 0.9, 0.5, 0.2])

print("tokens da resposta:", [t for t, m in zip(tokens, mascara) if m])
print("perda média sem máscara:", round(float(perdas.mean()), 3))
print("perda média com máscara (SFT):", round(float((perdas * mascara).sum() / mascara.sum()), 3))


# %% [bloco 2]
from sklearn.linear_model import LogisticRegression

def sig(x):
    return 1 / (1 + np.exp(-x))

def gerar_pares(n, peso_comprimento, rng):
    """Cada resposta tem 3 características: correta (0/1), cita fonte (0/1), comprimento (0 a 1)."""
    def feats(m):
        return np.column_stack([rng.random(m) < 0.6, rng.random(m) < 0.4, rng.random(m)]).astype(float)
    A, B = feats(n), feats(n)
    w = np.array([2.0, 1.0, peso_comprimento])          # recompensa verdadeira (oculta)
    a_vence = rng.random(n) < sig(A @ w - B @ w)         # Bradley-Terry
    return np.where(a_vence[:, None], A, B), np.where(a_vence[:, None], B, A)

rng = np.random.default_rng(0)
print(f"{'cenário':<32} {'coef. correta':>14} {'coef. cita':>11} {'coef. comprimento':>18} {'acurácia (pares de teste)':>26}")
for nome, pl in [("avaliadores penalizam extensão", -0.5), ("avaliadores preferem extensão", 0.8)]:
    W, L = gerar_pares(4000, pl, rng)
    D = W - L
    X = np.vstack([D, -D]); y = np.r_[np.ones(len(D)), np.zeros(len(D))]      # simetriza: (W-L)->1, (L-W)->0
    rm = LogisticRegression(fit_intercept=False, C=100).fit(X, y)
    Wt, Lt = gerar_pares(2000, pl, rng)
    acc = float((rm.decision_function(Wt - Lt) > 0).mean())
    c = rm.coef_[0]
    print(f"{nome:<32} {c[0]:>14.2f} {c[1]:>11.2f} {c[2]:>18.2f} {acc:>26.3f}")
print("recompensa verdadeira: correta = 2,0 | cita = 1,0 | comprimento = -0,5 ou +0,8")


# %% [bloco 3]
pref = np.array([0.50, 0.30, 0.15, 0.05])
r = np.array([0.0, 1.0, 2.0, 3.0])
print(f"{'beta':>5} | {'política ótima π*':<30} | {'KL(π*||ref)':>11} | {'recompensa esperada':>19}")
for beta in (0.1, 0.5, 1.0, 5.0):
    q = pref * np.exp(r / beta); q /= q.sum()
    kl = float((q * np.log(q / pref)).sum())
    print(f"{beta:>5} | {str(q.round(3)):<30} | {kl:>11.3f} | {float(q @ r):>19.3f}")
print(f"referência (β → ∞): recompensa esperada = {float(pref @ r):.3f}")


# %% [bloco 4]
def log_softmax(z):
    z = z - z.max()
    return z - np.log(np.exp(z).sum())

rng = np.random.default_rng(0)
N = 4000
a, b = rng.integers(0, 4, N), rng.integers(0, 4, N)
m = a != b
a, b = a[m], b[m]
vence_a = rng.random(len(a)) < sig(r[a] - r[b])           # Bradley-Terry com recompensas reais
W, L = np.where(vence_a, a, b), np.where(vence_a, b, a)   # índices das respostas preferida e rejeitada

beta = 1.0
lref = np.log(pref)
theta = lref.copy()                                        # a política começa igual à referência
I = np.eye(4)

def perda_dpo(theta):
    lp = log_softmax(theta)
    h = beta * ((lp[W] - lref[W]) - (lp[L] - lref[L]))
    return float(-np.log(sig(h)).mean())

print("perda inicial (política = referência):", round(perda_dpo(theta), 4), "| ln 2 =", round(float(np.log(2)), 4))
for passo in range(3000):
    lp = log_softmax(theta)
    h = beta * ((lp[W] - lref[W]) - (lp[L] - lref[L]))
    grad = (-(1 - sig(h))[:, None] * beta * (I[W] - I[L])).mean(axis=0)
    theta -= 1.0 * grad
politica = np.exp(log_softmax(theta))
alvo = pref * np.exp(r / beta); alvo /= alvo.sum()
print("perda final:", round(perda_dpo(theta), 4))
print("política aprendida pelo DPO:", politica.round(3))
print("política ótima do RLHF (π*): ", alvo.round(3))
print("diferença máxima:", round(float(np.abs(politica - alvo).max()), 3))

