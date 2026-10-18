"""Capítulo 6: soluções dos exercícios práticos e do projeto. Cada "# %%" é uma célula independente."""

# %% [Exercício 6.1: Comparando estratégias de decodificação]
import numpy as np
from llmdidatico import corpus, modelos

frases = corpus.frases_lm(3000, seed=0)
treino = frases[:2400]
lm = modelos.NgramaLM(n=3, k=0.05).ajustar(treino)
V = lm.vocab
vistas = {" ".join(f) for f in treino}

def gerar(rng, T=1.0, top_k=None, top_p=None, guloso=False, maxlen=14):
    ctx = []
    for _ in range(maxlen):
        lp = np.log(lm.distribuicao(ctx) + 1e-12)
        i = int(np.argmax(lp)) if guloso else modelos.amostrar(lp, rng, T=T, top_k=top_k, top_p=top_p)
        if V[i] == "<s>":
            continue
        ctx.append(V[i])
        if V[i] == ".":
            break
    return ctx

configs = {"gulosa": dict(guloso=True), "T=0,7": dict(T=0.7), "T=1,0": dict(T=1.0), "T=1,3": dict(T=1.3),
           "top-k=3": dict(top_k=3), "top-p=0,9": dict(top_p=0.9)}
print(f"{'estratégia':<11} {'semente':>7} {'válidas':>8} {'distintas':>10} {'no treino':>10}")
for nome, c in configs.items():
    for semente in (0, 1):
        rng = np.random.default_rng(semente)
        am = [gerar(rng, **c) for _ in range(300)]
        val = np.mean([corpus.frase_valida(s) for s in am])
        dist = len({" ".join(s) for s in am}) / 300
        no_treino = np.mean([" ".join(s) in vistas for s in am])
        print(f"{nome:<11} {semente:>7} {val:>8.2f} {dist:>10.2f} {no_treino:>10.2f}")


# %% [Exercício 6.2: Simulando latência e o efeito de limitar a saída]
import numpy as np

rng = np.random.default_rng(0)
N = 10_000
l_ent = np.maximum(1, rng.lognormal(np.log(800), 0.6, N).round()).astype(int)
l_sai = np.maximum(1, rng.lognormal(np.log(250), 0.7, N).round()).astype(int)

def latencia(l_ent, l_sai, prefill_tps=5000, s_por_token=0.03):
    return l_ent / prefill_tps + l_sai * s_por_token

def resumo(nome, t):
    p50, p95, p99 = np.percentile(t, [50, 95, 99])
    print(f"{nome:<22} média {t.mean():6.2f} s | p50 {p50:6.2f} | p95 {p95:6.2f} | p99 {p99:6.2f}")

resumo("sem limite de saída", latencia(l_ent, l_sai))
limitado = np.minimum(l_sai, 300)
resumo("limite de 300 tokens", latencia(l_ent, limitado))
print(f"respostas cortadas pelo limite: {np.mean(l_sai > 300):.1%}")
print(f"mediana do prompt: {np.median(l_ent):.0f} tokens | mediana da resposta: {np.median(l_sai):.0f} tokens")


# %% [Exercício 6.3: O cache KV: igualdade e economia]
import time
import numpy as np
from sklearn.linear_model import LinearRegression

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

d = 64
rng = np.random.default_rng(0)
WQ, WK, WV = (rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(3))

def sem_cache(X):
    out = []
    for t in range(1, len(X) + 1):
        pref = X[:t]
        Q, K, V = pref @ WQ, pref @ WK, pref @ WV
        out.append(softmax(Q[-1] @ K.T / np.sqrt(d)) @ V)
    return np.array(out)

def com_cache(X):
    Kc = np.zeros((len(X), d)); Vc = np.zeros((len(X), d)); out = []
    for t, x in enumerate(X):
        Kc[t] = x @ WK; Vc[t] = x @ WV
        out.append(softmax((x @ WQ) @ Kc[:t + 1].T / np.sqrt(d)) @ Vc[:t + 1])
    return np.array(out)

def mediana_tempo(f, X, rep=3):
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter(); f(X); ts.append(time.perf_counter() - t0)
    return float(np.median(ts))

Ts = [64, 128, 256, 512]
tA, tB = [], []
print(f"{'T':>5} {'iguais':>7} {'sem cache (ms)':>15} {'com cache (ms)':>15} {'razão':>7}")
for T in Ts:
    X = rng.normal(size=(T, d))
    igual = np.allclose(sem_cache(X), com_cache(X))
    tA.append(mediana_tempo(sem_cache, X)); tB.append(mediana_tempo(com_cache, X))
    print(f"{T:>5} {str(igual):>7} {1000 * tA[-1]:>15.1f} {1000 * tB[-1]:>15.1f} {tA[-1] / tB[-1]:>7.1f}")

for nome, ts in (("sem cache", tA), ("com cache", tB)):
    a = LinearRegression().fit(np.log(Ts).reshape(-1, 1), np.log(ts)).coef_[0]
    print(f"expoente de T ({nome}): {a:.2f}")


# %% [Projeto: Painel de decisão: que parâmetros de geração usar?]
import numpy as np
from llmdidatico import corpus, modelos, avaliacao as av

frases = corpus.frases_lm(3000, seed=0)
lm = modelos.NgramaLM(n=3, k=0.05).ajustar(frases[:2400])
V = lm.vocab

def gerar(rng, T, top_p, maxlen=14):
    ctx = []
    for _ in range(maxlen):
        lp = np.log(lm.distribuicao(ctx) + 1e-12)
        i = modelos.amostrar(lp, rng, T=T, top_p=None if top_p >= 1.0 else top_p)
        if V[i] == "<s>":
            continue
        ctx.append(V[i])
        if V[i] == ".":
            break
    return ctx

linhas = []
print(f"{'T':>4} {'top-p':>6} {'validade':>9} {'IC95% (validade)':>20} {'diversidade':>12}")
for T in (0.3, 0.5, 0.7, 1.0, 1.3):
    for tp in (1.0, 0.9, 0.8):
        rng = np.random.default_rng(0)
        am = [gerar(rng, T, tp) for _ in range(300)]
        v = np.array([float(corpus.frase_valida(s)) for s in am])
        m, lo, hi = av.bootstrap_ic(v, n=1000, seed=0)
        div = len({" ".join(s) for s in am}) / 300
        linhas.append((T, tp, m, lo, hi, div))
        print(f"{T:>4} {tp:>6} {m:>9.3f} {f'[{lo:.3f}; {hi:.3f}]':>20} {div:>12.2f}")

A = [l for l in linhas if l[3] >= 0.99]
B = [l for l in linhas if l[3] >= 0.95 and l[5] >= 0.70]
print("\nEquipe A (margem inferior da validade >= 0,99): ", [(l[0], l[1]) for l in A])
if A: print("  recomendada (menor diversidade, mais previsível):", min(A, key=lambda l: l[5])[:2])
print("Equipe B (validade >= 0,95 e diversidade >= 0,70):", [(l[0], l[1]) for l in B])
if B: print("  recomendada (maior diversidade):", max(B, key=lambda l: l[5])[:2])

