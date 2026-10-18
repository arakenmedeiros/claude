"""Capítulo 15: soluções dos exercícios práticos e do projeto. Cada "# %%" é uma célula independente."""

# %% [Exercício 15.1: Orçamento de custo e latência de um assistente]
P_ENT, P_SAI, P_CACHE = 1.0, 5.0, 0.1          # u.m. por milhão de tokens (HIPOTÉTICOS)
T_FIXO, T_VAR = 1200, 150
PREFILL_TPS, MS_TOKEN = 5000, 30
PEDIDOS = 200_000

def avaliar(t_saida, cache):
    p_fixo = P_CACHE if cache else P_ENT
    custo = (T_FIXO * p_fixo + T_VAR * P_ENT + t_saida * P_SAI) / 1e6
    tokens_prefill = T_VAR if cache else T_FIXO + T_VAR
    lat = tokens_prefill / PREFILL_TPS + t_saida * MS_TOKEN / 1000
    return custo, lat

configs = [("(a) base", 180, False), ("(b) saída 100", 100, False), ("(c) prefixo em cache", 180, True), ("(d) as duas", 100, True)]
base_c, base_l = avaliar(180, False)
print(f"{'configuração':<22} {'custo/dia (u.m.)':>17} {'relativo':>9} {'latência (s)':>13} {'relativa':>9}")
for nome, ts, ca in configs:
    c, l = avaliar(ts, ca)
    print(f"{nome:<22} {c*PEDIDOS:>17.1f} {c/base_c:>9.2f} {l:>13.2f} {l/base_l:>9.2f}")


# %% [Exercício 15.2: Sensibilidade de cada matriz à quantização]
import numpy as np
from llmdidatico import corpus
from llmdidatico.modelos import MLPLM

def reconstruir(W, bits):
    qmax = 2 ** (bits - 1) - 1
    s = np.abs(W).max(axis=0, keepdims=True) / qmax
    s = np.where(s == 0, 1.0, s)
    return np.clip(np.round(W / s), -qmax, qmax) * s

frases = corpus.frases_lm(3000, seed=2); teste = corpus.frases_lm(600, seed=3)
vocab = sorted({t for f in frases for t in f} | {"<s>"})
lm = MLPLM(vocab, ctx=3, d=16, h=48, seed=0); lm.treinar(frases, epocas=8)
P0 = {k: v.copy() for k, v in lm.P.items()}
base = lm.perplexidade(teste)
print(f"perplexidade original: {base:.4f}")

def ppl_config(bits_por_matriz):
    for k in ("E", "W1", "W2"):
        lm.P[k] = reconstruir(P0[k], bits_por_matriz[k]) if bits_por_matriz.get(k) else P0[k].copy()
    r = lm.perplexidade(teste)
    for k in P0: lm.P[k] = P0[k].copy()
    return r

tam = {k: P0[k].size for k in ("E", "W1", "W2")}
print(f"{'matriz':<7} {'parâmetros':>10} {'4 bits':>9} {'3 bits':>9}")
sens = {}
for k in ("E", "W1", "W2"):
    p4, p3 = ppl_config({k: 4}), ppl_config({k: 3})
    sens[k] = p3
    print(f"{k:<7} {tam[k]:>10} {p4:>9.4f} {p3:>9.4f}")
mais_sensivel = max(sens, key=sens.get)
print("mais sensível a 3 bits:", mais_sensivel)

def bits_medios(cfg):
    return sum(tam[k] * cfg[k] for k in tam) / sum(tam.values())

uniforme4 = {"E": 4, "W1": 4, "W2": 4}
mista = {k: (8 if k == mais_sensivel else 3) for k in tam}
for nome, cfg in (("uniforme 4 bits", uniforme4), ("mista (sensível 8, demais 3)", mista)):
    print(f"{nome:<30} perplexidade {ppl_config(cfg):.4f} | bits médios por parâmetro {bits_medios(cfg):.2f}")


# %% [Exercício 15.3: Escolhendo o limiar de um cache semântico]
import numpy as np
from llmdidatico import corpus
from llmdidatico.modelos import LSAEmbedder

X, y = corpus.intencoes(100, seed=9)
ENT = ["fatura", "nota fiscal", "boleto", "serviço", "aplicativo", "contrato", "internet", "1042", "8873", "2201", "5560", "premium", "básico", "família", "empresarial"]
entidades = lambda t: tuple(e for e in ENT if e in t.lower())
Xc, yc, Xt, yt = X[:400], y[:400], X[400:], y[400:]
emb = LSAEmbedder(dim=30).ajustar(Xc)

def simular(Xs, ys, tau):
    Z = emb.transformar(Xs); cz, ck, hit, erro = [], [], 0, 0
    for t, r, z in zip(Xs, ys, Z):
        k = (r, entidades(t))
        if cz:
            s = np.array(cz) @ z; j = int(s.argmax())
            if s[j] >= tau:
                hit += 1; erro += ck[j] != k; continue
        cz.append(z); ck.append(k)
    return hit / len(Xs), erro / max(hit, 1)

print(f"{'tau':>5} | {'calib: acerto':>13} {'erro':>6} | {'teste: acerto':>13} {'erro':>6}")
res = {}
for tau in (0.99, 0.95, 0.90, 0.80):
    res[tau] = (simular(Xc, yc, tau), simular(Xt, yt, tau))
    (hc, ec), (ht, et) = res[tau]
    print(f"{tau:>5.2f} | {hc:>13.3f} {ec:>6.3f} | {ht:>13.3f} {et:>6.3f}")
ok = [t for t in res if res[t][0][1] <= 0.15]
if ok:
    t_esc = min(ok); print(f"\nlimiar escolhido na calibração (erro <= 15%): {t_esc}; erro no teste: {res[t_esc][1][1]:.3f}")
else:
    print("\nnenhum limiar atinge erro <= 15% na calibração; o menor erro observado foi", round(min(res[t][0][1] for t in res), 3))


# %% [Projeto: Plano de capacidade, custo e metas de um assistente]
import numpy as np

T0, T1, T_HIT, META_P95, CUSTO_MODELO = 1.0, 0.1, 0.01, 3.0, 10.0       # parâmetros HIPOTÉTICOS

def simular(lam, B, h, n=30_000, seed=0):
    rng = np.random.default_rng(seed)
    chegada = np.cumsum(rng.exponential(1 / lam, n)); hit = rng.random(n) < h
    lat = np.full(n, T_HIT); idx = np.where(~hit)[0]; a = chegada[idx]; m = len(a); livre, i = 0.0, 0
    while i < m:
        ini = max(livre, a[i]); j = i
        while j < m and a[j] <= ini and j - i < B:
            j += 1
        livre = ini + T0 + T1 * (j - i); lat[idx[i:j]] = livre - a[i:j]; i = j
    return lat[n // 20:]

lams = np.round(np.arange(0.2, 6.01, 0.2), 1)
cap = {}
print(f"{'B':>3} {'h':>5} {'maior carga com p95 <= 3 s (req/s)':>38}")
for B in (1, 4, 8):
    for h in (0.0, 0.3):
        # a maior carga em que a meta vale para todas as cargas menores (a curva é crescente)
        maior = 0.0
        for l in lams:
            if np.quantile(simular(l, B, h), 0.95) <= META_P95: maior = l
            else: break
        cap[(B, h)] = maior
        print(f"{B:>3} {h:>5.1f} {maior:>38.1f}")

LAM = 3.0
print(f"\ncarga esperada: {LAM} req/s")
cands = []
for (B, h), maior in cap.items():
    if maior >= LAM:
        p95 = np.quantile(simular(LAM, B, h), 0.95)
        cands.append((CUSTO_MODELO * (1 - h) * 1000, B, h, p95))
cands.sort()
for custo, B, h, p95 in cands:
    print(f"B={B} h={h:.1f}: custo por 1000 pedidos = {custo:.0f} u.m.; p95 = {p95:.2f} s ({p95/META_P95:.0%} da meta)")
escolha = cands[0]
print("escolha:", f"B={escolha[1]}, h={escolha[2]}")
PICO = 5.0
p95_pico = np.quantile(simular(PICO, escolha[1], escolha[2]), 0.95)
print(f"no pico de {PICO} req/s, a configuração escolhida tem p95 = {p95_pico:.2f} s ({'cumpre' if p95_pico <= META_P95 else 'NÃO cumpre'} a meta)")

