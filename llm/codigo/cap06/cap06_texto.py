"""Capítulo 6: blocos de código do texto, em ordem. Cada "# %%" é uma célula."""

# %% [bloco 1]
import numpy as np
from llmdidatico import corpus, modelos

frases = corpus.frases_lm(3000, seed=0)
lm = modelos.NgramaLM(n=3, k=0.05).ajustar(frases[:2400])
V = lm.vocab

def gerar(lm, rng, T=1.0, top_k=None, top_p=None, guloso=False, maxlen=14):
    contexto = []
    for _ in range(maxlen):
        logp = np.log(lm.distribuicao(contexto) + 1e-12)
        i = int(np.argmax(logp)) if guloso else modelos.amostrar(logp, rng, T=T, top_k=top_k, top_p=top_p)
        tok = V[i]
        if tok == "<s>":
            continue
        contexto.append(tok)
        if tok == ".":
            break
    return contexto

rng = np.random.default_rng(0)
configs = {"gulosa": dict(guloso=True), "T=0,5": dict(T=0.5), "T=1,0": dict(T=1.0), "T=1,5": dict(T=1.5),
           "T=2,0": dict(T=2.0), "top-k=3": dict(top_k=3), "top-p=0,9": dict(top_p=0.9)}
print(f"{'estratégia':<11} {'válidas':>8} {'distintas':>10} {'lnP/token':>10}")
resumo = {}
for nome, c in configs.items():
    amostras = [gerar(lm, rng, **c) for _ in range(300)]
    validas = np.mean([corpus.frase_valida(s) for s in amostras])
    distintas = len({" ".join(s) for s in amostras}) / 300
    lp = np.mean([lm.logprob_frase(s) / len(s) for s in amostras])
    resumo[nome] = (validas, distintas, lp)
    print(f"{nome:<11} {validas:>8.2f} {distintas:>10.2f} {lp:>10.2f}")
print("\nexemplos com T=1,5:")
for s in [gerar(lm, rng, T=1.5) for _ in range(3)]:
    print("  ", " ".join(s))


# %% [bloco 2]
import math

def busca_feixe(lm, B=3, maxlen=10):
    feixe = [([], 0.0)]            # (sequência, log-probabilidade acumulada)
    finais = []
    for _ in range(maxlen):
        candidatos = []
        for seq, lp in feixe:
            p = lm.distribuicao(seq)
            for i in np.argsort(-p)[:B]:                # expande só os B melhores tokens
                tok = V[i]
                if tok == "<s>":
                    continue
                cand = (seq + [tok], lp + math.log(p[i]))
                (finais if tok == "." else candidatos).append(cand)
        feixe = sorted(candidatos, key=lambda x: -x[1])[:B]
        if not feixe:
            break
    return sorted(finais, key=lambda x: -x[1])

guloso = gerar(lm, np.random.default_rng(0), guloso=True)
print("gulosa:", " ".join(guloso), f"| lnP total = {lm.logprob_frase(guloso):.2f}")
for B in (1, 3, 5):
    melhor = busca_feixe(lm, B=B)[0]
    print(f"feixe B={B}:", " ".join(melhor[0]), f"| lnP total = {melhor[1]:.2f}")


# %% [bloco 3]
def orcamento(janela, instrucoes, historico, documentos, saida_max):
    usado = instrucoes + historico + documentos
    livre = janela - usado
    return {"entrada": usado, "livre": livre, "cabe_saida": livre >= saida_max, "saida_possivel": max(0, min(livre, saida_max))}

print(orcamento(janela=8192, instrucoes=300, historico=2500, documentos=4000, saida_max=1000))
print(orcamento(janela=8192, instrucoes=300, historico=2500, documentos=5200, saida_max=1000))


# %% [bloco 4]
def latencia(l_entrada, l_saida, prefill_tps=5000, ms_por_token=30):
    t_prefill = l_entrada / prefill_tps
    t_decode = l_saida * ms_por_token / 1000
    return t_prefill, t_decode, t_prefill + t_decode

for le, ls in [(1000, 200), (2000, 200), (1000, 400), (8000, 100)]:
    tp, td, tt = latencia(le, ls)
    print(f"entrada {le:>5} | saída {ls:>4} | prefill {tp:5.2f} s | decodificação {td:5.2f} s | total {tt:5.2f} s")


# %% [bloco 5]
import time

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

d = 64
rng = np.random.default_rng(0)
WQ, WK, WV = (rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(3))
X = rng.normal(size=(256, d))                       # embeddings de 256 tokens (já "gerados")

def sem_cache(X):
    saidas = []
    for t in range(1, len(X) + 1):
        pref = X[:t]
        Q, K, V = pref @ WQ, pref @ WK, pref @ WV   # recalcula tudo a cada passo
        a = softmax(Q[-1] @ K.T / np.sqrt(d))
        saidas.append(a @ V)
    return np.array(saidas)

def com_cache(X):
    Kc, Vc, saidas = [], [], []
    for x in X:
        Kc.append(x @ WK); Vc.append(x @ WV)         # só o token novo
        q = x @ WQ
        a = softmax(q @ np.array(Kc).T / np.sqrt(d))
        saidas.append(a @ np.array(Vc))
    return np.array(saidas)

t0 = time.perf_counter(); A = sem_cache(X); t_sem = time.perf_counter() - t0
t0 = time.perf_counter(); B = com_cache(X); t_com = time.perf_counter() - t0
print("saídas idênticas (sem x com cache):", np.allclose(A, B))
print(f"tempo sem cache: {t_sem * 1000:.1f} ms | com cache: {t_com * 1000:.1f} ms | razão: {t_sem / t_com:.1f}x")


# %% [bloco 6]
def memoria_kv_gb(camadas, T, n_kv, d_cab, bytes_num=2):
    return 2 * camadas * T * n_kv * d_cab * bytes_num / 1e9

# modelo hipotético: 32 camadas, 32 cabeças de 128 dimensões, float16
print(f"MHA  (32 cabeças KV), T=4096 : {memoria_kv_gb(32, 4096, 32, 128):.2f} GB por sequência")
print(f"GQA  ( 8 cabeças KV), T=4096 : {memoria_kv_gb(32, 4096, 8, 128):.2f} GB por sequência")
print(f"MHA  (32 cabeças KV), T=32768: {memoria_kv_gb(32, 32768, 32, 128):.2f} GB por sequência")
print(f"10 usuários simultâneos (MHA, T=4096): {10 * memoria_kv_gb(32, 4096, 32, 128):.1f} GB só de cache")


# %% [bloco 7]
a, b = np.float32(1e8), np.float32(5.0)
print("(a + b) + b =", (a + b) + b, "| a + (b + b) =", a + (b + b), "(iguais em matemática exata, diferentes em float32)")

x = np.random.default_rng(0).normal(size=1_000_000).astype(np.float32)
print("soma float32 em ordem direta :", x.sum(), "| em ordem aleatória:", x[np.random.default_rng(1).permutation(len(x))].sum())

def amostrar_frase(semente):
    return " ".join(gerar(lm, np.random.default_rng(semente), T=1.0))

print("\nmesma semente  :", amostrar_frase(7) == amostrar_frase(7))
print("sementes diferentes:", [amostrar_frase(s) for s in (1, 2, 3)])

