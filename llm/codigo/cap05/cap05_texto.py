"""Capítulo 5: blocos de código do texto, em ordem. Cada "# %%" é uma célula."""

# %% [bloco 1]
import numpy as np
from llmdidatico import corpus, modelos

frases = corpus.frases_lm(3000, seed=0)
frase = frases[0]
print("frase:", " ".join(frase))

# 1) próximo token: pares (contexto, alvo)
print("\npares (contexto -> alvo) para o treino causal:")
for t in range(1, 5):
    print(f"  {' '.join(frase[:t]):<28} -> {frase[t]}")

# 2) mascaramento: esconde ~15% dos tokens (aqui, escolhidos por semente)
rng = np.random.default_rng(0)
mascarados = rng.random(len(frase)) < 0.25
entrada = ["[MASK]" if m else t for t, m in zip(frase, mascarados)]
alvos = [t for t, m in zip(frase, mascarados) if m]
print("\nmascaramento:\n  entrada:", " ".join(entrada), "\n  alvos  :", alvos)

# 3) perda de uma frase com o modelo de trigramas
lm = modelos.NgramaLM(n=3, k=0.05).ajustar(frases[:2400])
lp = lm.logprob_frase(frase)
print(f"\nperda por token de '{' '.join(frase)}' = {-lp / len(frase):.3f} nats")


# %% [bloco 2]
def taxa(s, S, eta_max=1e-3, eta_min=1e-4, w=100):
    if s <= w:
        return eta_max * s / w
    return eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * (s - w) / (S - w)))

def recortar(grad, norma_max=1.0):
    n = np.linalg.norm(grad)
    return grad * (norma_max / n) if n > norma_max else grad

S = 1000
for s in (1, 50, 100, 500, 1000):
    print(f"passo {s:>4}: taxa = {taxa(s, S):.6f}")
g = np.array([3.0, 4.0])
print("gradiente original:", g, "| norma =", np.linalg.norm(g), "| recortado:", recortar(g).round(3), "| norma =", np.linalg.norm(recortar(g)).round(3))


# %% [bloco 3]
import time
frases = corpus.frases_lm(12000, seed=0)
validacao = frases[-1000:]
treino = frases[:400]
vocab = sorted({t for f in frases for t in f} | {"<s>"})

m = modelos.MLPLM(vocab, ctx=3, d=16, h=64, seed=0)
print("parâmetros do modelo:", sum(p.size for p in m.P.values()))
t0 = time.perf_counter()
print(f"{'época':>5} {'perda treino':>13} {'perda validação':>16}")
for ep in range(1, 13):
    h = m.treinar(treino, epocas=1, lote=64, lr=0.01, seed=ep)
    X, y = m.dados(validacao); _, _, z = m.frente(X)
    lp = modelos.log_softmax(z)
    perda_val = float(-lp[np.arange(len(y)), y].mean())
    print(f"{ep:>5} {h[0]:>13.3f} {perda_val:>16.3f}")
print(f"tempo de treino: {time.perf_counter() - t0:.2f} s | perplexidade de validação: {m.perplexidade(validacao):.2f}")
tri = modelos.NgramaLM(n=3, k=0.05).ajustar(frases[:10000])
print(f"referência: trigramas com 10 000 frases -> perplexidade {tri.perplexidade(validacao):.2f}")


# %% [bloco 4]
print(f"{'frases de treino':>17} {'perda treino':>13} {'perda validação':>16}")
resultados = []
for n in (20, 50, 100, 200, 400, 800, 1600, 3200):
    mm = modelos.MLPLM(vocab, ctx=3, d=16, h=64, seed=0)
    epocas = int(max(6, min(60, 18000 / n)))
    mm.treinar(frases[:n], epocas=epocas, lote=64, lr=0.01)
    pt = np.log(mm.perplexidade(frases[:min(n, 400)])); pv = np.log(mm.perplexidade(validacao))
    resultados.append((n, pt, pv))
    print(f"{n:>17} {pt:>13.3f} {pv:>16.3f}")


# %% [bloco 5]
def shingles(texto, k=3):
    p = texto.lower().split()
    return {tuple(p[i:i + k]) for i in range(len(p) - k + 1)}

def jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else 0.0

docs = [
    "a biblioteca central funciona de segunda a sexta das 7h às 22h e aos sábados das 8h às 14h",
    "a biblioteca central funciona de segunda a sexta das 7h às 22h e aos sábados das 8h às 14h",   # cópia exata
    "a biblioteca central funciona de segunda a sexta das 7h às 23h e aos sábados das 8h às 14h",   # quase cópia
    "o restaurante universitário serve almoço e jantar com preços diferentes para quem tem bolsa",
]
S = [shingles(d) for d in docs]
print("Jaccard entre pares de documentos:")
for i in range(len(docs)):
    for j in range(i + 1, len(docs)):
        print(f"  ({i},{j}): {jaccard(S[i], S[j]):.2f}")

manter = []
for i, s in enumerate(S):
    if all(jaccard(s, S[j]) < 0.5 for j in manter):
        manter.append(i)
print("documentos mantidos com limiar 0,5:", manter)


# %% [bloco 6]
def flops_treino(N, D):
    return 6 * N * D

def dias(flops, flops_por_s_por_acelerador, n_aceleradores, eficiencia=1.0):
    return flops / (flops_por_s_por_acelerador * n_aceleradores * eficiencia) / 86400

N, D = 7e9, 1e12
C = flops_treino(N, D)
print(f"FLOPs de treino: {C:.2e}")
for n_acel in (1, 100, 1000):
    print(f"{n_acel:>5} aceleradores a 1e14 FLOP/s, eficiência 100%: {dias(C, 1e14, n_acel):>10.1f} dias | com 40% de eficiência: {dias(C, 1e14, n_acel, 0.4):>10.1f} dias")

def memoria_treino_gb(N, bytes_por_parametro=16):
    return N * bytes_por_parametro / 1e9
print(f"memória de treino (≈16 bytes/parâmetro) para N={N:.0e}: {memoria_treino_gb(N):.0f} GB, sem contar ativações")
print(f"tokens por parâmetro: {D / N:.0f}")


# %% [bloco 7]
def ngramas(tokens, n=4):
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}

from llmdidatico import DOCS
treino_ng = set()
for d in DOCS[:50]:                                          # só 50 documentos entram no "treino"
    treino_ng |= ngramas(modelos.tokenizar(d[3]), 4)

print(f"{'documento':<10} {'sobreposição de 4-gramas com o treino':>40}")
for d in DOCS[45:55]:
    ng = ngramas(modelos.tokenizar(d[3]), 4)
    sobre = len(ng & treino_ng) / len(ng)
    marca = "  <- no treino" if d in DOCS[:50] else "  <- fora do treino"
    print(f"{d[0]:<10} {sobre:>40.2f}{marca}")

