"""Capítulo 15: blocos de código do texto, em ordem. Cada "# %%" é uma célula."""

# %% [bloco 1]
import numpy as np

# HARDWARE E MODELO HIPOTÉTICOS (valores didáticos; não descrevem nenhum produto real)
BW = 2.0e12        # largura de banda da memória: bytes por segundo
PICO = 3.0e14      # capacidade de cálculo: operações de ponto flutuante por segundo
N = 7e9            # parâmetros do modelo
KV_TOKEN = 2 * 32 * 32 * 128 * 2   # bytes de cache por token (fórmula do Cap. 6: 32 camadas, 32 cabeças de 128, float16)

def t_passo(b, T_ctx, bytes_param=2.0):
    """Tempo (s) de um passo de decodificação com b sequências simultâneas, cada uma com T_ctx tokens de contexto."""
    t_mem = (N * bytes_param + b * T_ctx * KV_TOKEN) / BW     # ler pesos (uma vez) + cache de cada sequência
    t_calc = b * 2 * N / PICO                                  # ~2N operações por token gerado
    return t_mem, t_calc

print(f"{'lote':>5} {'t_mem (ms)':>11} {'t_calc (ms)':>12} {'gargalo':>9} {'tokens/s (total)':>17} {'tokens/s (por usuário)':>23}")
for b in (1, 8, 32, 128, 256):
    tm, tc = t_passo(b, T_ctx=1000)
    t = max(tm, tc)
    print(f"{b:>5} {tm*1e3:>11.1f} {tc*1e3:>12.2f} {'memória' if tm > tc else 'cálculo':>9} {b/t:>17.0f} {1/t:>23.1f}")

print()
tm, tc = t_passo(1, 1000, bytes_param=0.5)
tm16, _ = t_passo(1, 1000, bytes_param=2.0)
print(f"lote 1, pesos em 4 bits: t_mem = {tm*1e3:.2f} ms contra {tm16*1e3:.2f} ms em float16 (razão {tm16/tm:.2f}x)")

def t_prefill(L, bytes_param=2.0):
    return max(N * bytes_param / BW, 2 * N * L / PICO)
for L in (50, 500, 2000):
    print(f"prefill de {L:>4} tokens: {t_prefill(L)*1e3:7.1f} ms | {'cálculo' if 2*N*L/PICO > N*2/BW else 'memória'}")


# %% [bloco 2]
# PREÇOS HIPOTÉTICOS, em "unidades monetárias" (u.m.) por milhão de tokens. Não são preços de nenhum serviço.
PRECO_ENTRADA, PRECO_SAIDA, PRECO_ENTRADA_CACHE = 1.0, 5.0, 0.1

def custo_req(t_fixo, t_var, t_saida, prefixo_em_cache=False):
    """t_fixo: tokens do prefixo repetido (instruções, exemplos); t_var: tokens da parte variável; t_saida: tokens gerados."""
    p_fixo = PRECO_ENTRADA_CACHE if prefixo_em_cache else PRECO_ENTRADA
    return (t_fixo * p_fixo + t_var * PRECO_ENTRADA + t_saida * PRECO_SAIDA) / 1e6

cenarios = {
    "base (fixo 800, variável 200, saída 150)": dict(t_fixo=800, t_var=200, t_saida=150),
    "saída limitada a 90 tokens":               dict(t_fixo=800, t_var=200, t_saida=90),
    "prefixo de 800 tokens em cache":           dict(t_fixo=800, t_var=200, t_saida=150, prefixo_em_cache=True),
    "RAG: +600 tokens de contexto":             dict(t_fixo=800, t_var=800, t_saida=150),
    "RAG + prefixo em cache + saída 90":        dict(t_fixo=800, t_var=800, t_saida=90, prefixo_em_cache=True),
}
base = custo_req(**cenarios["base (fixo 800, variável 200, saída 150)"])
print(f"{'cenário':<44} {'u.m. / 1000 req':>16} {'relativo':>9}")
for nome, kw in cenarios.items():
    c = custo_req(**kw)
    print(f"{nome:<44} {c*1000:>16.3f} {c/base:>9.2f}")
req_dia = 100_000
print(f"\nbase: {base*req_dia:.1f} u.m. por dia para {req_dia:,} requisições/dia ({base*req_dia*30:.0f} u.m. em 30 dias)".replace(",", " "))


# %% [bloco 3]
def quantizar(W, bits=8, eixo=None, grupo=None):
    """Quantização simétrica uniforme: devolve (inteiros, escala). Valores reconstruídos = inteiros * escala."""
    qmax = 2 ** (bits - 1) - 1
    if grupo:                                   # escala por grupo de `grupo` valores consecutivos
        Wg = W.reshape(-1, grupo)
        s = np.abs(Wg).max(axis=1, keepdims=True) / qmax
    elif eixo is None:                          # uma escala para o tensor inteiro
        Wg, s = W, np.abs(W).max() / qmax
    else:                                       # uma escala por coluna (eixo=0) ou por linha (eixo=1)
        Wg, s = W, np.abs(W).max(axis=eixo, keepdims=True) / qmax
    s = np.where(s == 0, 1.0, s)
    q = np.clip(np.round(Wg / s), -qmax, qmax)
    return q, s

def reconstruir(W, bits=8, **kw):
    q, s = quantizar(W, bits, **kw)
    return (q * s).reshape(W.shape)

def erro_rel(W, Wq):
    return float(np.sqrt(np.mean((W - Wq) ** 2)) / np.sqrt(np.mean(W ** 2)))

rng = np.random.default_rng(0)
W = rng.normal(0, 0.02, (256, 512))             # pesos sintéticos "bem comportados"
Wo = W.copy(); Wo[:, ::64] *= 30                # as mesmas matrizes, com 8 colunas de valores atípicos (outliers)
print(f"{'pesos':<10} {'bits':>4} {'por tensor':>11} {'por coluna':>11} {'por grupo de 64':>16}   (erro relativo, RMSE/RMS)")
for nome, w in (("normais", W), ("outliers", Wo)):
    for bits in (8, 6, 4, 3, 2):
        e = [erro_rel(w, reconstruir(w, bits, **kw)) for kw in ({}, {"eixo": 0}, {"grupo": 64})]
        print(f"{nome:<10} {bits:>4} {e[0]:>11.4f} {e[1]:>11.4f} {e[2]:>16.4f}")

q, s = quantizar(W, 8, eixo=0)
q8 = q.astype(np.int8)
print(f"\narmazenamento: float64 = {W.nbytes} bytes | int8 + escalas float32 = {q8.nbytes + s.astype(np.float32).nbytes} bytes")
x = rng.normal(size=(4, 256))
y_ref, y_q = x @ W, (x @ q8.astype(float)) * s
print("erro relativo da saída da camada (int8 por coluna):", round(float(np.linalg.norm(y_ref - y_q) / np.linalg.norm(y_ref)), 4))


# %% [bloco 4]
from llmdidatico import corpus
from llmdidatico.modelos import MLPLM

frases = corpus.frases_lm(3000, seed=0); teste = corpus.frases_lm(600, seed=1)
vocab = sorted({t for f in frases for t in f} | {"<s>"})
lm = MLPLM(vocab, ctx=3, d=16, h=48, seed=0); lm.treinar(frases, epocas=8)
P0 = {k: v.copy() for k, v in lm.P.items()}
n_par = sum(v.size for v in P0.values())
ppl0 = lm.perplexidade(teste)
print(f"MLPLM: {n_par} parâmetros | perplexidade no teste (float64): {ppl0:.4f}")

def com_pesos_quantizados(bits, **kw):
    for k in ("E", "W1", "W2"):                  # os vieses ficam em precisão cheia (são poucos)
        lm.P[k] = reconstruir(P0[k], bits, **kw)
    return lm.perplexidade(teste)

print(f"{'bits':>4} {'por tensor':>11} {'por coluna':>11}   (perplexidade; referência {ppl0:.3f})")
ppl_q = {}
for bits in (8, 6, 4, 3, 2):
    a, b = com_pesos_quantizados(bits), com_pesos_quantizados(bits, eixo=0)
    ppl_q[bits] = (a, b)
    print(f"{bits:>4} {a:>11.3f} {b:>11.3f}")
for k in P0: lm.P[k] = P0[k].copy()
print("pesos restaurados; perplexidade:", round(lm.perplexidade(teste), 4))


# %% [bloco 5]
import numpy as np
from llmdidatico import avaliacao as av
from llmdidatico.modelos import softmax

def espiral(n, seed, ruido=0.08, k=3, voltas=4.0):
    r = np.random.default_rng(seed); t = r.random(n); c = r.integers(0, k, n)
    ang = voltas * t + c * 2 * np.pi / k; rad = 0.2 + t
    return np.c_[rad * np.cos(ang), rad * np.sin(ang)] + r.normal(0, ruido, (n, 2)), c

class MLP:
    """Rede de uma camada oculta (tanh) com gradientes manuais; alvo = distribuição (rótulo duro ou suave)."""
    def __init__(s, h, seed=0, k=3):
        r = np.random.default_rng(seed)
        s.W1, s.b1 = r.normal(0, .7, (2, h)), np.zeros(h); s.W2, s.b2 = r.normal(0, .7, (h, k)), np.zeros(k)
    def f(s, X):
        a = np.tanh(X @ s.W1 + s.b1); return a, a @ s.W2 + s.b2
    def treinar(s, X, alvo, ep=3000, lr=0.5, T=1.0):
        for _ in range(ep):
            a, z = s.f(X); dz = (softmax(z, T=T) - alvo) / len(X); da = dz @ s.W2.T * (1 - a ** 2)
            s.W1 -= lr * X.T @ da; s.b1 -= lr * da.sum(0); s.W2 -= lr * a.T @ dz; s.b2 -= lr * dz.sum(0)
        return s
    def prever(s, X): return s.f(X)[1].argmax(1)
    def n_par(s): return s.W1.size + s.b1.size + s.W2.size + s.b2.size

T_DEST = 2.0
linhas = []
for semente in range(8):
    X, y = espiral(200, semente); Xt, yt = espiral(4000, 100 + semente); Xu, _ = espiral(3000, 500 + semente)
    prof = MLP(64, semente).treinar(X, np.eye(3)[y], ep=6000)                 # professor: 64 unidades, rótulos verdadeiros
    suave = softmax(prof.f(Xu)[1], T=T_DEST)                                  # saídas suaves do professor em dados SEM rótulo
    aluno_r = MLP(12, semente).treinar(X, np.eye(3)[y], ep=6000)               # aluno treinado só com os 200 rótulos
    aluno_d = MLP(12, semente).treinar(Xu, suave, ep=3000, T=T_DEST)           # aluno treinado com o professor (destilação)
    linhas.append([(m.prever(Xt) == yt).mean() for m in (prof, aluno_r, aluno_d)])
linhas = np.array(linhas)
for nome, col in zip(("professor (387 parâmetros)", "aluno com rótulos (75)", "aluno destilado (75)"), linhas.T):
    print(f"{nome:<28} acurácia = {col.mean():.4f}  (desvio {col.std():.4f}, 8 sementes)")
m, lo, hi = av.bootstrap_ic(linhas[:, 2] - linhas[:, 1], n=2000, seed=0)
print(f"diferença pareada destilado - rótulos: {m:+.4f}  IC95% [{lo:+.4f}; {hi:+.4f}]")
print("parâmetros:", prof.n_par(), "(professor) e", aluno_d.n_par(), "(aluno)")


# %% [bloco 6]
# Mistura de especialistas (MoE): um roteador escolhe k de E especialistas para cada token
rng = np.random.default_rng(0)
T, d, E, k = 4096, 32, 8, 2
x = rng.normal(size=(T, d))
Wr = rng.normal(0, 0.5, (d, E)); Wr[:, 0] += 0.25             # roteador sintético, ligeiramente "viciado" no especialista 0
p = softmax(x @ Wr)
escolhidos = np.argsort(-p, axis=1)[:, :k]
carga = np.bincount(escolhidos.ravel(), minlength=E) / (T * k)
print("fração de tokens (x k) por especialista:", np.round(carga, 3))
print(f"carga máxima / carga ideal (1/E): {carga.max() * E:.2f}  -> o especialista mais usado limita a velocidade do grupo")

# parâmetros totais x ativos (configuração HIPOTÉTICA)
compartilhado, por_especialista = 1.0e9, 0.5e9
total = compartilhado + E * por_especialista
ativo = compartilhado + k * por_especialista
print(f"total {total/1e9:.1f} bi | ativo por token {ativo/1e9:.1f} bi ({ativo/total:.0%}) | mas a memória precisa guardar os {total/1e9:.1f} bi")


# %% [bloco 7]
import numpy as np
from llmdidatico import corpus
def memoria_kv_gb(camadas, T, n_kv, d_cab, bytes_num=2):     # fórmula da Seção 6.9.3
    return 2 * camadas * T * n_kv * d_cab * bytes_num / 1e9

# Cache de prefixo: o prefixo (instruções + exemplos) é igual em todas as requisições
def prefill_s(L, tps=5000):  return L / tps                   # modelo do Cap. 6: 5 000 tokens/s (hipotético)
L_pref, L_var = 2000, 150
sem = prefill_s(L_pref + L_var)
for taxa_acerto in (0.0, 0.5, 0.9, 1.0):
    # com acerto, só a parte variável é processada; no erro, processa tudo (e grava o prefixo)
    medio = taxa_acerto * prefill_s(L_var) + (1 - taxa_acerto) * sem
    print(f"acerto do cache de prefixo {taxa_acerto:>4.0%}: TTFT médio {medio*1e3:6.1f} ms ({medio/sem:5.2f} do TTFT sem cache)")

print(f"\ncache KV de um prefixo de {L_pref} tokens (32 camadas, 32 cabeças, float16): {memoria_kv_gb(32, L_pref, 32, 128):.2f} GB")
print("se 50 prefixos diferentes ficarem retidos:", round(50 * memoria_kv_gb(32, L_pref, 32, 128), 1), "GB")


# %% [bloco 8]
import re
from llmdidatico.modelos import LSAEmbedder

Xc, yc = corpus.intencoes(100, seed=5)              # 800 mensagens de atendimento (sintéticas)
ENTIDADES = ["fatura", "nota fiscal", "boleto", "serviço", "aplicativo", "contrato", "internet",
             "1042", "8873", "2201", "5560", "premium", "básico", "família", "empresarial"]

def entidades(texto):
    t = texto.lower()
    return tuple(e for e in ENTIDADES if e in t)

def chave_correta(texto, rotulo):                   # a "resposta certa" depende da intenção E das entidades
    return (rotulo, entidades(texto))

# 1) cache exato: mesma mensagem (após normalizar espaços e caixa)
vistos, acertos = set(), 0
for t in Xc:
    k = " ".join(t.lower().split())
    acertos += k in vistos; vistos.add(k)
print(f"cache exato: taxa de acerto = {acertos/len(Xc):.3f}")

# 2) cache semântico: reutiliza a resposta da mensagem mais parecida, se cosseno >= tau
Zc = LSAEmbedder(dim=30).ajustar(Xc).transformar(Xc)

def cache_semantico(tau, exigir_entidades=False):
    cz, ck, ce, hit, erro = [], [], [], 0, 0
    for t, r, z in zip(Xc, yc, Zc):
        k = chave_correta(t, r)
        if cz:
            s = np.array(cz) @ z; j = int(s.argmax())
            if s[j] >= tau and (not exigir_entidades or ce[j] == k[1]):
                hit += 1; erro += ck[j] != k; continue
        cz.append(z); ck.append(k); ce.append(k[1])
    return hit / len(Xc), erro / max(hit, 1)

print(f"\n{'tau':>5} | {'acerto':>7} {'erro entre os acertos':>22} | {'com verificação de entidades: acerto':>37} {'erro':>6}")
res_cache = {}
for tau in (0.99, 0.95, 0.90, 0.80, 0.60):
    h, e = cache_semantico(tau); h2, e2 = cache_semantico(tau, True)
    res_cache[tau] = (h, e, h2, e2)
    print(f"{tau:>5.2f} | {h:>7.3f} {e:>22.3f} | {h2:>37.3f} {e2:>6.3f}")


# %% [bloco 9]
rng = np.random.default_rng(0)
L_saida = np.clip(rng.geometric(1 / 60, size=400), 5, 400)    # comprimentos de saída (tokens) de 400 pedidos
print(f"comprimento médio {L_saida.mean():.1f} tokens | máximo {L_saida.max()}")

def lote_estatico(L, B):
    """Cada grupo de B pedidos ocupa o servidor até terminar o MAIS LONGO."""
    return int(sum(L[i:i + B].max() for i in range(0, len(L), B)))

def lote_continuo(L, B):
    """Ao terminar um pedido, outro entra no lugar na hora (a cada passo de decodificação)."""
    fila, ativos, passos = list(L[::-1]), [], 0
    while fila or ativos:
        while fila and len(ativos) < B:
            ativos.append(fila.pop())
        passos += 1
        ativos = [r - 1 for r in ativos if r - 1 > 0]
    return passos

util = L_saida.sum()
print(f"{'B':>3} {'passos estático':>16} {'passos contínuo':>16} {'uso estático':>13} {'uso contínuo':>13} {'ganho':>6}")
res_lote = {}
for B in (1, 4, 8, 16, 32):
    e, c = lote_estatico(L_saida, B), lote_continuo(L_saida, B)
    res_lote[B] = (e, c, util / (e * B), util / (c * B))
    print(f"{B:>3} {e:>16} {c:>16} {util/(e*B):>13.3f} {util/(c*B):>13.3f} {e/c:>5.2f}x")


# %% [bloco 10]
import math

def erlang_c(c, a):
    """Probabilidade de esperar numa fila M/M/c, com carga oferecida a = lambda/mu < c."""
    soma = sum(a ** k / math.factorial(k) for k in range(c))
    termo = a ** c / (math.factorial(c) * (1 - a / c))
    return termo / (soma + termo)

def teoria_mmc(lam, mu, c):
    pw = erlang_c(c, lam / mu); wq = pw / (c * mu - lam)
    return pw, wq, wq + 1 / mu

def simular_mmc(lam, mu, c, n=100_000, seed=0):
    rng = np.random.default_rng(seed)
    chegada = np.cumsum(rng.exponential(1 / lam, n)); servico = rng.exponential(1 / mu, n)
    livre = np.zeros(c); espera = np.empty(n); lat = np.empty(n)
    for i in range(n):
        j = livre.argmin(); ini = max(chegada[i], livre[j]); livre[j] = ini + servico[i]
        espera[i], lat[i] = ini - chegada[i], livre[j] - chegada[i]
    k = n // 20                                  # descarta o aquecimento
    return espera[k:], lat[k:]

mu = 1.0      # cada servidor atende 1 requisição por segundo (hipotético)
print(f"{'c':>2} {'lambda':>6} {'rho':>5} | {'P(esperar) teo':>14} {'sim':>6} | {'W médio teo':>11} {'sim':>6} | {'p95 sim':>8}")
for c, lam in ((1, 0.5), (1, 0.8), (1, 0.9), (4, 2.0), (4, 3.2), (4, 3.6)):
    pw, wq, w = teoria_mmc(lam, mu, c); esp, lat = simular_mmc(lam, mu, c)
    print(f"{c:>2} {lam:>6.1f} {lam/(c*mu):>5.2f} | {pw:>14.3f} {(esp>1e-12).mean():>6.3f} | {w:>11.2f} {lat.mean():>6.2f} | {np.quantile(lat,.95):>8.2f}")


# %% [bloco 11]
def simular_lote(lam, B, T0=1.0, T1=0.1, h=0.0, n=40_000, seed=0, t_hit=0.01):
    """Um servidor que processa até B requisições por vez; um lote com b itens dura T0 + T1*b segundos (hipotético).
    Requisições que acertam o cache (fração h) são respondidas em t_hit e não entram na fila."""
    rng = np.random.default_rng(seed)
    chegada = np.cumsum(rng.exponential(1 / lam, n)); hit = rng.random(n) < h
    lat = np.full(n, t_hit); idx = np.where(~hit)[0]; a = chegada[idx]; m = len(a)
    livre, i = 0.0, 0
    while i < m:
        ini = max(livre, a[i]); j = i
        while j < m and a[j] <= ini and j - i < B:
            j += 1
        livre = ini + T0 + T1 * (j - i); lat[idx[i:j]] = livre - a[i:j]; i = j
    return lat[n // 20:]

configs = [("B=1", 1, 0.0), ("B=1 + cache 40%", 1, 0.4), ("B=8", 8, 0.0), ("B=8 + cache 40%", 8, 0.4)]
print(f"{'lambda (req/s)':>14} | " + " | ".join(f"{nome:^17}" for nome, _, _ in configs) + "   (média / p95, em segundos)")
res_fila = {}
for lam in (0.3, 0.6, 0.8, 0.9, 1.5, 3.0):
    cel = []
    for nome, B, h in configs:
        l = simular_lote(lam, B, h=h); res_fila[(nome, lam)] = (l.mean(), np.quantile(l, .95))
        cel.append(f"{l.mean():>7.2f} / {np.quantile(l,.95):>6.2f}")
    print(f"{lam:>14.1f} | " + " | ".join(f"{c:^17}" for c in cel))
print("capacidade máxima sem lote: 1/(T0+T1) =", round(1 / 1.1, 3), "req/s | com B=8:", round(8 / (1.0 + 0.8), 3), "req/s")


# %% [bloco 12]
from llmdidatico.modelos import NgramaLM

def tokens_esperados(alfa, gama):
    """Esperança de tokens produzidos por rodada (Leviathan et al., 2023): (1 - a^(g+1)) / (1 - a)."""
    return (1 - alfa ** (gama + 1)) / (1 - alfa)

def ganho_tempo(alfa, gama, c):
    """Fator de aceleração em tempo: tokens por rodada / custo da rodada (g rascunhos de custo c + 1 verificação)."""
    return tokens_esperados(alfa, gama) / (gama * c + 1)

def monte_carlo(alfa, gama, n=200_000, seed=0):
    aceitos = np.random.default_rng(seed).random((n, gama)) < alfa
    k = np.where(aceitos.all(1), gama, np.argmin(aceitos, 1))     # nº de rascunhos aceitos em sequência
    return (k + 1).mean()                                         # + 1 token do modelo grande (correção ou bônus)

print(f"{'alfa':>5} {'gama':>5} {'fórmula':>8} {'Monte Carlo':>12}")
for a, g in ((0.5, 4), (0.8, 4), (0.9, 4), (0.8, 8)):
    print(f"{a:>5.1f} {g:>5} {tokens_esperados(a, g):>8.3f} {monte_carlo(a, g):>12.3f}")

c = 0.05      # custo do rascunhador relativo ao modelo grande (hipotético)
print(f"\nganho de tempo com c = {c} (linhas: alfa; colunas: gama)")
print("       " + "".join(f"{g:>7}" for g in (1, 2, 4, 8, 16)))
for a in (0.5, 0.7, 0.8, 0.9, 0.95):
    print(f"{a:>5.2f}  " + "".join(f"{ganho_tempo(a, g, c):>7.2f}" for g in (1, 2, 4, 8, 16)))

# Experimento: alvo = trigramas, rascunhador = bigramas, no corpus sintético. Verifica que a distribuição da saída NÃO muda.
fr = corpus.frases_lm(3000, seed=0)
alvo, rasc = NgramaLM(3, 0.01).ajustar(fr), NgramaLM(2, 0.01).ajustar(fr)
V = alvo.vocab
def dist(m, ctx):
    p = m.distribuicao(ctx); return p / p.sum()

def gerar_alvo(rng, maxlen=14):
    s = []
    while len(s) < maxlen:
        t = V[rng.choice(len(V), p=dist(alvo, s))]; s.append(t)
        if t == ".": break
    return s

def gerar_especulativo(rng, gama=4, maxlen=14):
    s, rodadas, aceitos, propostos = [], 0, 0, 0
    while len(s) < maxlen and (not s or s[-1] != "."):
        rodadas += 1; ctx = list(s); rasc_tok, rasc_q = [], []
        for _ in range(gama):                                   # o rascunhador propõe até gama tokens
            q = dist(rasc, ctx); t = int(rng.choice(len(V), p=q))
            rasc_tok.append(t); rasc_q.append(q); ctx.append(V[t])
            if V[t] == ".": break
        ctx = list(s); rejeitou = False
        for t, q in zip(rasc_tok, rasc_q):                      # o alvo verifica (na prática, tudo em uma só passada)
            p = dist(alvo, ctx); propostos += 1
            if rng.random() < min(1.0, p[t] / q[t]):
                aceitos += 1; ctx.append(V[t])
                if V[t] == ".": break
            else:                                               # rejeição: amostra da distribuição residual max(p - q, 0)
                r = np.maximum(p - q, 0); ctx.append(V[int(rng.choice(len(V), p=r / r.sum()))]); rejeitou = True; break
        else:
            if ctx[-1] != ".": ctx.append(V[int(rng.choice(len(V), p=dist(alvo, ctx)))])   # token bônus
        s = ctx[:maxlen]
    return s, rodadas, aceitos, propostos

rng = np.random.default_rng(0); N_AM = 800
A = [gerar_alvo(rng) for _ in range(N_AM)]
R = [gerar_especulativo(rng) for _ in range(N_AM)]
print(f"\nfrases gramaticais: alvo {np.mean([corpus.frase_valida(s) for s in A]):.3f} | especulativo {np.mean([corpus.frase_valida(r[0]) for r in R]):.3f}")
print(f"comprimento médio: alvo {np.mean([len(s) for s in A]):.2f} | especulativo {np.mean([len(r[0]) for r in R]):.2f}")
alfa_emp = sum(r[2] for r in R) / sum(r[3] for r in R)
print(f"taxa de aceitação medida: {alfa_emp:.3f} | tokens por rodada do alvo: {np.mean([len(r[0]) / r[1] for r in R]):.2f} (sem especulação: 1,00)")


# %% [bloco 13]
from llmdidatico import avaliacao as av
import json
from llmdidatico.simulador import SimuladorLLM, INTENCOES, _PALAVRAS, _sem_acentos
from sklearn.linear_model import LogisticRegression

V1 = ("cancelamento", "segunda_via", "suporte_tecnico")          # as únicas categorias que o simulador v1 "conhece"

def sinais(texto):
    """Sinais de confiança do modelo barato (programados por nós): acertos de palavras-chave dentro e fora do seu repertório."""
    t = _sem_acentos(texto); h = {k: sum(w in t for w in ws) for k, ws in _PALAVRAS.items()}
    p1 = sorted((h[k] for k in V1), reverse=True); fora = sum(h[k] for k in h if k not in V1)
    return [p1[0], p1[1], fora, len(t.split())]

def rodar(semente, n=60):
    X, y = corpus.intencoes(n, seed=semente); barato, caro = SimuladorLLM("v1", 0), SimuladorLLM("v3", 0); b = []
    for t in X:
        r = barato.responder(t); ach = [c for c in INTENCOES if c.replace("_", " ") in r]; b.append(ach[0] if ach else "indefinido")
    c = [json.loads(caro.responder(t))["intencao"] for t in X]
    return X, np.array(y), np.array(b), np.array(c)

Xa, ya, ba, ca = rodar(61); Xt, yt, bt, ct = rodar(62)             # roteador treina em uma amostra e é avaliado em outra
roteador = LogisticRegression(max_iter=1000).fit([sinais(t) for t in Xa], ba == ya)
conf = roteador.predict_proba([sinais(t) for t in Xt])[:, 1]       # confiança = P(modelo barato acertar)
ok_b, ok_c = bt == yt, ct == yt
print(f"acerto isolado: barato {ok_b.mean():.3f} | caro {ok_c.mean():.3f}")

CUSTO_BARATO, CUSTO_CARO = 1.0, 10.0                                # u.m. por requisição (HIPOTÉTICO)
rng = np.random.default_rng(0)
print(f"\n{'limiar':>7} {'escalonadas':>12} {'acerto':>7} {'acerto (sorteio)':>17} {'custo/req':>10} {'custo relativo ao caro':>23}")
res_rot = {}
for tau in (0.0, 0.3, 0.6, 0.8, 0.95, 1.01):
    esc = conf < tau; acc = np.where(esc, ok_c, ok_b).mean()
    acc_sorteio = np.mean([np.where(rng.random(len(esc)) < esc.mean(), ok_c, ok_b).mean() for _ in range(300)])
    custo = CUSTO_BARATO + CUSTO_CARO * esc.mean()                  # o barato sempre roda; o caro só nos escalonados
    res_rot[tau] = (esc.mean(), acc, custo)
    print(f"{tau:>7.2f} {esc.mean():>12.3f} {acc:>7.3f} {acc_sorteio:>17.3f} {custo:>10.2f} {custo/CUSTO_CARO:>23.2f}")
esc = conf < 0.6
print(f"\nentre as {(~esc).sum()} mensagens que o roteador manteve no modelo barato (limiar 0,6), o barato acertou {ok_b[~esc].mean():.3f}")
print(f"entre as {esc.sum()} escalonadas, o barato sozinho acertaria {ok_b[esc].mean():.3f} e o caro acerta {ok_c[esc].mean():.3f}")


# %% [bloco 14]
# Combinando alavancas (todos os preços e custos são HIPOTÉTICOS; as taxas vêm dos experimentos acima)
h_cache = res_cache[0.95][2]          # acerto do cache semântico com verificação de entidades (tau = 0,95)
frac_esc = res_rot[0.6][0]            # fração escalonada ao modelo caro (limiar 0,6)
estrategias = {
    "A: sempre o modelo caro":                      CUSTO_CARO,
    "B: roteamento (limiar 0,6)":                   CUSTO_BARATO + CUSTO_CARO * frac_esc,
    "C: cache semântico + modelo caro":             (1 - h_cache) * CUSTO_CARO,
    "D: cache semântico + roteamento":              (1 - h_cache) * (CUSTO_BARATO + CUSTO_CARO * frac_esc),
}
print(f"taxa de acerto do cache = {h_cache:.3f} | fração escalonada = {frac_esc:.3f}")
for nome, c in estrategias.items():
    print(f"{nome:<36} custo médio por requisição = {c:5.2f} u.m. | por 1 000 requisições = {c*1000:8.1f} | relativo a A = {c/CUSTO_CARO:.2f}")


# %% [bloco 15]
class BaldeDeFichas:
    """Limitador de taxa por balde de fichas: repõe `taxa` fichas por segundo, até `capacidade`. O relógio é simulado."""
    def __init__(self, taxa, capacidade):
        self.taxa, self.capacidade, self.fichas, self.t = taxa, capacidade, capacidade, 0.0
    def permitir(self, agora, custo=1.0):
        self.fichas = min(self.capacidade, self.fichas + (agora - self.t) * self.taxa); self.t = agora
        if self.fichas >= custo:
            self.fichas -= custo; return True, 0.0
        return False, (custo - self.fichas) / self.taxa            # (recusada, segundos sugeridos até tentar de novo)

rng = np.random.default_rng(3)
chegadas = np.cumsum(rng.exponential(1 / 8.0, 400))                 # carga média de 8 req/s, com rajadas
chegadas = chegadas[chegadas < 30]
balde = BaldeDeFichas(taxa=5.0, capacidade=10)                      # limite: 5 req/s sustentadas, rajada de até 10
aceitas, recusadas, esperas = 0, 0, []
for t in chegadas:
    ok, espera = balde.permitir(t)
    aceitas += ok; recusadas += (not ok)
    if not ok: esperas.append(espera)
print(f"requisições em 30 s: {len(chegadas)} | aceitas {aceitas} ({aceitas/30:.2f}/s) | recusadas {recusadas} ({recusadas/len(chegadas):.1%})")
print(f"espera sugerida às recusadas: mediana {np.median(esperas):.2f} s")
rajada = BaldeDeFichas(5.0, 10)
print("rajada de 15 requisições no mesmo instante:", sum(rajada.permitir(0.0)[0] for _ in range(15)), "aceitas")


# %% [bloco 16]
import json, time
from scipy import stats

# 1) Log estruturado: um registro JSON por chamada (sem o texto do usuário: privacidade)
registro = {"ts": "2026-10-18T12:00:00Z", "id_req": "r-000123", "prompt_versao": "triagem-v3", "modelo": "modelo-A@2026-09",
            "tokens_entrada": 1010, "tokens_saida": 87, "ttft_ms": 210, "latencia_ms": 2650, "cache": "prefixo", "status": "ok", "custo_um": 0.00048}
print(json.dumps(registro, ensure_ascii=False))

# 2) Percentis de latência (a média esconde a cauda). Dados: simulação de fila da seção anterior
lat = simular_lote(0.9, 8, h=0.4)
p = np.quantile(lat, [.5, .95, .99])
print(f"\nlatência: média {lat.mean():.2f} s | p50 {p[0]:.2f} | p95 {p[1]:.2f} | p99 {p[2]:.2f}")

# 3) SLO e orçamento de erro
slo, janela_min = 0.995, 30 * 24 * 60
print(f"SLO de {slo:.1%} em 30 dias -> orçamento de indisponibilidade = {(1 - slo) * janela_min:.0f} minutos")
LIMITE_S = 2.0
print(f"fração das requisições simuladas com latência <= {LIMITE_S:.0f} s (meta: 99%): {(lat <= LIMITE_S).mean():.3f}")

# 4) Deriva: PSI sobre a distribuição de intenções e teste KS sobre o comprimento das mensagens
def psi(ref, atual, rotulos):
    e = np.array([max(np.mean(np.array(ref) == r), 1e-4) for r in rotulos]); a = np.array([max(np.mean(np.array(atual) == r), 1e-4) for r in rotulos])
    return float(np.sum((a - e) * np.log(a / e)))

X_ref, y_ref = corpus.intencoes(100, seed=71)
X_new, y_new = corpus.intencoes(100, seed=72)
pesos = np.array([3.0 if r == "reclamacao" else 1.0 for r in y_new]); rng = np.random.default_rng(0)
sel0 = rng.choice(len(y_new), 800, replace=True)                                  # produção sem mudança
sel = rng.choice(len(y_new), 800, replace=True, p=pesos / pesos.sum())          # a produção passou a ter 3x mais reclamações
X_der, y_der = [X_new[i] for i in sel], [y_new[i] for i in sel]
rotulos = sorted(set(y_ref))
print(f"\nPSI (sem deriva, outra amostra da mesma distribuição): {psi(y_ref, [y_new[i] for i in sel0], rotulos):.4f}")
print(f"PSI (reclamações 3x mais frequentes):                  {psi(y_ref, y_der, rotulos):.4f}")
len_ref = [len(t.split()) for t in X_ref]; len_der = [len(t.split()) + int(rng.integers(0, 9)) for t in X_der[:400]]   # mensagens ficaram mais longas
print(f"KS (comprimento): sem deriva p = {stats.ks_2samp(len_ref, [len(t.split()) for t in X_new]).pvalue:.3f} | com deriva p = {stats.ks_2samp(len_ref, len_der).pvalue:.2e}")

# 5) Rastro (API do OpenTelemetry; sem SDK configurado, o rastreador é um 'não faz nada' seguro)
from opentelemetry import trace
rastreador = trace.get_tracer("app.atendimento")
with rastreador.start_as_current_span("chamar_modelo") as span:
    span.set_attribute("prompt.versao", "triagem-v3"); span.set_attribute("tokens.entrada", 1010)
    print("\nspan gravando?", span.is_recording())


# %% [bloco 17]
# Energia por requisição: E = P x t. Potência e tempos HIPOTÉTICOS; o objetivo é o raciocínio, não o valor.
P_W = 700.0                           # potência média do acelerador durante a decodificação (W), hipotética
t_passo_s = 0.04                      # duração de um passo de decodificação (s), hipotética
def energia_wh(tokens_saida, lote):
    t = tokens_saida * t_passo_s                       # o acelerador fica ocupado por t segundos...
    return P_W * t / lote / 3600                       # ...dividido entre as `lote` requisições que o compartilham
for lote in (1, 8, 32):
    e = energia_wh(200, lote)
    print(f"lote {lote:>2}: {e:6.3f} Wh por requisição de 200 tokens | {e*1e6/1000:8.1f} kWh por milhão de requisições")
print("\nresposta 30% mais curta (140 tokens), lote 8:", round(energia_wh(140, 8), 3), "Wh; fator", round(energia_wh(140, 8) / energia_wh(200, 8), 2))
print("(supõe potência constante; na prática ela varia com a carga, e o consumo total inclui refrigeração e rede)")

