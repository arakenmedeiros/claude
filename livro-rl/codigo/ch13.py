"""
Livro: Aprendizado por Reforço — Sistemas Multiagente, RLHF e Fronteiras
Autor do livro: Araken de Medeiros Santos

Os blocos abaixo seguem a ordem do texto e devem ser executados em sequência (como células de um notebook).
Os trechos marcados como 'não executado' (por exemplo, downloads do Kaggle) aparecem comentados.
"""
import warnings; warnings.filterwarnings('ignore')

# ======================================================================
# Bloco 1
# ======================================================================
import numpy as np

PAGAMENTOS = np.array([[(3, 3), (0, 5)], [(5, 0), (1, 1)]])          # ação 0 = cooperar, 1 = trair

def dilema_iterado(n_rodadas=20_000, alpha=0.1, gamma=0.9, eps_ini=0.3, semente=0):
    rng = np.random.default_rng(semente); Q = [np.zeros((5, 2)), np.zeros((5, 2))]     # estado 0 = início; 1..4 = (minha, sua) últimas ações
    estado = [0, 0]; coop = []
    for t in range(n_rodadas):
        eps = max(0.02, eps_ini * (1 - t / (0.7 * n_rodadas)))
        acoes = [int(rng.integers(2)) if rng.random() < eps else int(np.argmax(Q[i][estado[i]] + 1e-9 * rng.random(2))) for i in range(2)]
        r = PAGAMENTOS[acoes[0], acoes[1]]
        novo = [min(1 + 2 * acoes[i] + acoes[1 - i], 4) for i in range(2)]
        for i in range(2):
            Q[i][estado[i], acoes[i]] += alpha * (r[i] + gamma * Q[i][novo[i]].max() - Q[i][estado[i], acoes[i]])
        estado = novo; coop.append(1 - np.mean(acoes))                  # fração de cooperação nesta rodada
    return np.array(coop)

finais = np.array([dilema_iterado(semente=s)[-2000:].mean() for s in range(20)])
print("cooperação final por semente:", finais.round(2))
print(f"coopera (> 90%): {np.mean(finais > 0.9):.0%} | trai (< 10%): {np.mean(finais < 0.1):.0%} | outros: {np.mean((finais >= 0.1) & (finais <= 0.9)):.0%}")

# ======================================================================
# Bloco 2
# ======================================================================
def cara_ou_coroa(n=30_000, alpha=0.05, T=0.3, semente=0):
    rng = np.random.default_rng(semente); Q = [np.zeros(2), np.zeros(2)]; P = []
    softmax = lambda q: np.exp((q - q.max()) / T) / np.exp((q - q.max()) / T).sum()
    for _ in range(n):
        p = [softmax(Q[0]), softmax(Q[1])]
        a = [int(rng.random() < p[0][1]), int(rng.random() < p[1][1])]
        r1 = 1.0 if a[0] == a[1] else -1.0
        Q[0][a[0]] += alpha * (r1 - Q[0][a[0]]); Q[1][a[1]] += alpha * (-r1 - Q[1][a[1]])
        P.append((p[0][1], p[1][1]))
    return np.array(P)

P = cara_ou_coroa()
print(f"média de P(coroa): jogador 1 = {P[:, 0].mean():.3f} | jogador 2 = {P[:, 1].mean():.3f}   (equilíbrio de Nash: 0,5)")
print(f"oscilação (desvio-padrão): jogador 1 = {P[:, 0].std():.3f} | jogador 2 = {P[:, 1].std():.3f}")

# ======================================================================
# Bloco 3
# ======================================================================
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

vinho = fetch_ucirepo(id=186)
X = vinho.data.features.to_numpy(float); q = vinho.data.targets.iloc[:, 0].to_numpy(float)
Z = StandardScaler().fit_transform(X); col_alc = list(vinho.data.features.columns).index("alcohol")
Z_tr, Z_te, q_tr, q_te = train_test_split(Z, q, test_size=0.3, random_state=0)
rng = np.random.default_rng(0)

def gerar_pares(Zx, qx, n_pares, vies=1.0, tau=0.7):
    i = rng.integers(0, len(Zx), n_pares); j = rng.integers(0, len(Zx), n_pares)
    util = qx + vies * Zx[:, col_alc] * qx.std()                       # utilidade percebida: qualidade + viés pelo álcool
    p_i = 1 / (1 + np.exp(-(util[i] - util[j]) / tau))                  # Bradley–Terry
    return i, j, (rng.random(n_pares) < p_i).astype(int)                # 1 = anotador prefere o item i

def treinar_modelo_recompensa(i, j, y, C=1.0):
    D = Z_tr[i] - Z_tr[j]                                               # diferença de características do par
    mr = LogisticRegression(fit_intercept=False, C=C, max_iter=1000).fit(np.vstack([D, -D]), np.concatenate([y, 1 - y]))
    return mr.coef_[0]                                                  # r(x) = w · x

for vies in [0.0, 1.0]:
    i, j, y = gerar_pares(Z_tr, q_tr, 4000, vies=vies); w = treinar_modelo_recompensa(i, j, y)
    r_te = Z_te @ w
    print(f"\nviés dos anotadores = {vies} | correlação entre recompensa aprendida e qualidade real = {np.corrcoef(r_te, q_te)[0, 1]:.2f}")
    for N in [1, 4, 16, 64, 256]:
        ids = rng.integers(0, len(Z_te), (400, N)); melhor = ids[np.arange(400), np.argmax(r_te[ids], axis=1)]
        print(f"  melhor de {N:3d} candidatos | recompensa-proxy média = {r_te[melhor].mean():5.2f} | qualidade real média = {q_te[melhor].mean():.3f}")

# ======================================================================
# Bloco 4
# ======================================================================
import torch

def perda_dpo(logp_pref, logp_rej, ref_pref, ref_rej, beta=0.1):
    margem = beta * ((logp_pref - ref_pref) - (logp_rej - ref_rej))
    return -torch.nn.functional.logsigmoid(margem)

# log-probabilidades (invente números plausíveis) da resposta preferida e da rejeitada
ref_pref, ref_rej = torch.tensor(-20.0), torch.tensor(-22.0)         # modelo de referência
for nome, lp, lr_ in [("sem mudança", -20.0, -22.0), ("aumentou a preferida", -18.0, -22.0), ("aumentou a preferida e reduziu a rejeitada", -18.0, -26.0)]:
    L = perda_dpo(torch.tensor(lp), torch.tensor(lr_), ref_pref, ref_rej, beta=0.5)
    print(f"{nome:44s} | perda DPO = {L.item():.4f}")

# ======================================================================
# Bloco 5
# ======================================================================
import numpy as np

# Duas salas 9×9 separadas por uma parede na coluna 4, com uma porta em (linha 4, coluna 4). Início (0,0); meta (8,8).
N, PORTA, META = 9, (4, 4), (8, 8)
def livre(r, c): return 0 <= r < N and 0 <= c < N and (c != 4 or r == 4)
MOV = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def passo(s, a):
    r, c = s; dr, dc = MOV[a]; r2, c2 = r + dr, c + dc
    return (r2, c2) if livre(r2, c2) else s

def acao_para_porta(s):                                  # política INTERNA da opção "ir até a porta"
    r, c = s
    if r != PORTA[0]: return 1 if r < PORTA[0] else 0
    return 3 if c < PORTA[1] else 2

def acao_para_meta(s):                                   # política INTERNA da opção "ir até a meta"
    r, c = s
    return 3 if c < META[1] else 1

def opcao_valida(o, s):                                  # conjunto de iniciação
    if o < 4: return True
    if o == 4: return s != PORTA                         # "ir até a porta": de qualquer lugar, exceto da própria porta
    return s[1] >= 4 and s != META                       # "ir até a meta": só da porta ou da segunda sala

def executar_opcao(o, s, gamma, limite):
    """Devolve (novo estado, recompensa acumulada com desconto, duração em passos, terminou?)."""
    if o < 4: s2 = passo(s, o); return s2, -1.0, 1, s2 == META
    alvo = PORTA if o == 4 else META; pol = acao_para_porta if o == 4 else acao_para_meta
    R, k = 0.0, 0
    while s != alvo and k < limite:                     # a opção termina ao chegar ao alvo (β = 1 lá)
        s = passo(s, pol(s)); R += -1.0 * gamma ** k; k += 1
        if s == META: break
    return s, R, k, s == META

def treinar(com_opcoes, semente, n_ep=60, alpha=0.2, eps=0.1, gamma=0.99):
    rng = np.random.default_rng(semente); nO = 6 if com_opcoes else 4
    Q = np.zeros((N, N, nO)); passos_ep = []
    for ep in range(n_ep):
        s, total, fim = (0, 0), 0, False
        while not fim and total < 3000:
            validas = [o for o in range(nO) if opcao_valida(o, s)]
            if rng.random() < eps: o = int(rng.choice(validas))
            else: q = np.array([Q[s][o] if o in validas else -np.inf for o in range(nO)]); o = int(q.argmax())
            s2, R, k, fim = executar_opcao(o, s, gamma, 100)
            validas2 = [x for x in range(nO) if opcao_valida(x, s2)]
            alvo = R + (0 if fim else gamma ** k * max(Q[s2][x] for x in validas2))
            Q[s][o] += alpha * (alvo - Q[s][o]); s = s2; total += k
        passos_ep.append(total)
    return np.array(passos_ep)

for nome, flag in [("apenas ações primitivas", False), ("com opções (porta e meta)", True)]:
    P = np.array([treinar(flag, s) for s in range(20)])
    print(f"{nome:26s} | passos até a meta, média de 20 sementes — episódios 1, 5, 20, 60: {P.mean(axis=0)[[0, 4, 19, 59]].round(0)}")

# ======================================================================
# Bloco 6
# ======================================================================
import numpy as np

# Cada candidato a resposta tem: correta (sim/não; 30% de chance) e "estilo" (quão confiante e elaborada ela parece).
# Três "juízes" escolhem a melhor entre N candidatos (best-of-N). Medimos a taxa REAL de respostas corretas escolhidas.
rng = np.random.default_rng(0)
def taxa_real(juiz, N, n_prob=4000):
    ok = 0
    for _ in range(n_prob):
        correta = rng.random(N) < 0.30; estilo = rng.normal(size=N)
        escolhida = np.argmax(juiz(correta, estilo))
        ok += correta[escolhida]
    return ok / n_prob

verificador_exato   = lambda cor, est: cor + 1e-6 * rng.random(len(cor))                        # confere o resultado
verificador_falho   = lambda cor, est: np.where(cor, 1.0, (rng.random(len(cor)) < 0.2) * 1.0) + 1e-6 * rng.random(len(cor))  # aceita 20% das erradas
modelo_preferencia  = lambda cor, est: 0.5 * cor + 1.0 * est + 0.5 * rng.normal(size=len(cor))  # gosta de estilo confiante

print(f"{'N':>5} | {'verificador exato':>17} | {'verificador falho':>17} | {'modelo de preferência':>21}")
for N in (1, 4, 16, 64, 256):
    print(f"{N:5d} | {taxa_real(verificador_exato, N):17.3f} | {taxa_real(verificador_falho, N):17.3f} | {taxa_real(modelo_preferencia, N):21.3f}")

# ======================================================================
# Bloco 7
# ======================================================================
import numpy as np
from scipy.optimize import linprog

# Três ações com (recompensa, custo de segurança). Restrição: custo esperado por passo ≤ 0,6.
r = np.array([1.0, 2.0, 3.0]); c = np.array([0.0, 0.5, 1.5]); limite = 0.6

# 1) Sem restrição: escolhe a ação de maior recompensa
a = r.argmax(); print(f"sem restrição : ação {a} | recompensa = {r[a]:.2f} | custo = {c[a]:.2f}  (viola o limite de {limite})")

# 2) Solução exata do problema com restrição (programa linear sobre probabilidades das ações)
res = linprog(-r, A_ub=[c], b_ub=[limite], A_eq=[np.ones(3)], b_eq=[1], bounds=[(0, 1)] * 3)
print(f"ótimo restrito: π = {res.x.round(3)} | recompensa = {r @ res.x:.2f} | custo = {c @ res.x:.2f}")

# 3) Penalização fixa: maximiza r − λ·c com λ escolhido "no olho"
for lam in (0.0, 0.5, 1.0, 2.0):
    a = (r - lam * c).argmax(); print(f"penalização λ = {lam:3.1f} | ação {a} | recompensa = {r[a]:.2f} | custo = {c[a]:.2f}"
                                       f" | {'viola' if c[a] > limite else 'respeita'} o limite")

# 4) Método lagrangiano (subida dual): λ sobe quando o custo médio passa do limite e desce quando sobra folga
lam, eta, hist = 0.0, 0.05, []
for t in range(5000):
    a = (r - lam * c).argmax(); hist.append(a)               # melhor resposta ao λ atual
    lam = max(0.0, lam + eta * (c[a] - limite))
pi = np.bincount(hist, minlength=3) / len(hist)
print(f"lagrangiano   : π = {pi.round(3)} | recompensa = {r @ pi:.2f} | custo = {c @ pi:.2f} | λ final = {lam:.2f}")

# ======================================================================
# Bloco 8
# ======================================================================
from sklearn.utils import resample

i, j, y = gerar_pares(Z_tr, q_tr, 300, vies=1.0)
w_unico = treinar_modelo_recompensa(i, j, y)
conjunto = []
for s in range(10):
    k = resample(np.arange(len(y)), random_state=s)
    conjunto.append(treinar_modelo_recompensa(i[k], j[k], y[k]))
R = np.array([Z_te @ wk for wk in conjunto]); r_unico = Z_te @ w_unico

def qualidade_real(r, N, rep=300):
    ids = rng.integers(0, len(Z_te), (rep, N)); melhor = ids[np.arange(rep), np.argmax(r[ids], axis=1)]
    return q_te[melhor].mean()

for N in [4, 64, 1024]:
    linha = {"N": N, "modelo único": round(qualidade_real(r_unico, N), 3)}
    for lam in [0.0, 1.0, 2.0]:
        linha[f"conjunto λ={lam}"] = round(qualidade_real(R.mean(0) - lam * R.std(0), N), 3)
    print(linha)

