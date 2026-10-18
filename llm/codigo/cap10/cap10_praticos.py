"""Capítulo 10: soluções dos exercícios práticos e do projeto. Cada "# %%" é uma célula independente."""

# %% [Exercício 10.1: Normalização e a ordem dos resultados]
import numpy as np
from llmdidatico import DOCS, CONSULTAS, modelos, avaliacao as av
from llmdidatico.indices import pontuar

ids = [d[0] for d in DOCS]; textos = [d[2] + ". " + d[3] for d in DOCS]
lsa = modelos.LSAEmbedder(dim=40).ajustar(textos); Dn = lsa.transformar(textos)
compr = np.array([len(modelos.tokenizar_sem_stop(t)) for t in textos], float)
Du = Dn * compr[:, None]
COM = [c for c in CONSULTAS if c[1]]
Q = lsa.transformar([c[0] for c in COM])

print(f"{'versão':<16}{'medida':<12}{'MRR':>7}{'IC95%':>20}")
for nome, D in (("normalizada", Dn), ("não normalizada", Du)):
    for metrica in ("produto", "cosseno", "euclidiana"):
        S = pontuar(Q, D, metrica)
        mrrs = [av.mrr([ids[j] for j in np.argsort(-S[i], kind="stable")], c[1]) for i, c in enumerate(COM)]
        m, lo, hi = av.bootstrap_ic(mrrs, n=2000, seed=0)
        print(f"{nome:<16}{metrica:<12}{m:>7.3f}{f'[{lo:.3f}; {hi:.3f}]':>20}")


# %% [Exercício 10.2: Segmentação e o método de recuperação]
import numpy as np
from llmdidatico import DOCS, CONSULTAS, modelos, avaliacao as av
from llmdidatico.indices import chunk_palavras

ids = [d[0] for d in DOCS]; textos = [d[2] + ". " + d[3] for d in DOCS]; tok = modelos.tokenizar_sem_stop
inicio, pos = [], 0
for t in textos:
    n = len(t.split()); inicio.append((pos, pos + n)); pos += n
manual = " ".join(textos)
COM = [c for c in CONSULTAS if c[1]]
alvo = {}
for q, rel, ans, _ in COM:
    i = ids.index(rel[0]); k = textos[i].find(ans)
    w0 = inicio[i][0] + len(textos[i][:k].split()); alvo[q] = (w0, w0 + len(ans.split()))

print(f"{'tamanho':>8}{'pedaços':>9}{'método':>8}{'hit@3':>8}{'IC95%':>20}")
for tam in (20, 40, 80):
    ped = chunk_palavras(manual, tam, tam // 2)
    bm = modelos.BM25([tok(p[2]) for p in ped])
    lsa = modelos.LSAEmbedder(dim=40).ajustar([p[2] for p in ped]); E = lsa.transformar([p[2] for p in ped])
    for nome in ("BM25", "LSA"):
        acertos = []
        for q, (a0, a1) in alvo.items():
            s = bm.pontuar(tok(q)) if nome == "BM25" else E @ lsa.transformar([q])[0]
            top = np.argsort(-s, kind="stable")[:3]
            acertos.append(float(any(ped[j][0] <= a0 and ped[j][1] >= a1 for j in top)))
        m, lo, hi = av.bootstrap_ic(acertos, n=2000, seed=0)
        print(f"{tam:>8}{len(ped):>9}{nome:>8}{m:>8.3f}{f'[{lo:.3f}; {hi:.3f}]':>20}")


# %% [Exercício 10.3: O valor de k no RRF e o peso da fusão]
import numpy as np
from llmdidatico import DOCS, CONSULTAS, modelos, avaliacao as av
from llmdidatico.indices import rrf

ids = [d[0] for d in DOCS]; textos = [d[2] + ". " + d[3] for d in DOCS]; tok = modelos.tokenizar_sem_stop
bm = modelos.BM25([tok(t) for t in textos]); lsa = modelos.LSAEmbedder(dim=40).ajustar(textos); Dl = lsa.transformar(textos)
COM = [c for c in CONSULTAS if c[1]]
rk = lambda s: [ids[i] for i in np.argsort(-s, kind="stable")]
sb = {c[0]: bm.pontuar(tok(c[0])) for c in COM}; sl = {c[0]: Dl @ lsa.transformar([c[0]])[0] for c in COM}
nz = lambda s: (s - s.min()) / (s.max() - s.min() + 1e-12)
mrr_de = lambda f: np.array([av.mrr(f(c[0]), c[1]) for c in COM])

def dif(a, b, n=5000, seed=0):
    d = b - a; rng = np.random.default_rng(seed)
    med = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return d.mean(), np.quantile(med, 0.025), np.quantile(med, 0.975)

base = mrr_de(lambda q: rk(sb[q]))
print(f"BM25 isolado: MRR = {base.mean():.3f}")
print(f"{'configuração':<16}{'MRR':>7}{'dif. vs BM25':>14}{'IC95% da dif.':>20}")
for k in (1, 5, 10, 30, 60, 120):
    v = mrr_de(lambda q, k=k: rrf([rk(sb[q]), rk(sl[q])], k=k)[0]); d, lo, hi = dif(base, v)
    print(f"{f'RRF k={k}':<16}{v.mean():>7.3f}{d:>+14.3f}{f'[{lo:+.3f}; {hi:+.3f}]':>20}")
for a in (0.0, 0.25, 0.5, 0.75, 1.0):
    v = mrr_de(lambda q, a=a: rk(a * nz(sb[q]) + (1 - a) * nz(sl[q]))); d, lo, hi = dif(base, v)
    print(f"{f'soma alfa={a}':<16}{v.mean():>7.3f}{d:>+14.3f}{f'[{lo:+.3f}; {hi:+.3f}]':>20}")


# %% [Projeto: Buscador com recusa calibrada, reranqueamento e relatório com incerteza]
import numpy as np
from sklearn.linear_model import LogisticRegression
from llmdidatico import DOCS, CONSULTAS, modelos, avaliacao as av
from llmdidatico.indices import rrf

ids = [d[0] for d in DOCS]; textos = [d[2] + ". " + d[3] for d in DOCS]; tok = modelos.tokenizar_sem_stop
bm = modelos.BM25([tok(t) for t in textos]); lsa = modelos.LSAEmbedder(dim=40).ajustar(textos); Dl = lsa.transformar(textos)
titulos = [set(tok(d[2])) for d in DOCS]; vocab = [set(tok(t)) for t in textos]

def candidatos(q, K=10):
    sb = bm.pontuar(tok(q)); sl = Dl @ lsa.transformar([q])[0]
    rb = [ids[i] for i in np.argsort(-sb, kind="stable")]; rl = [ids[i] for i in np.argsort(-sl, kind="stable")]
    top = [ids.index(d) for d in rrf([rb, rl])[0][:K]]; qt = set(tok(q))
    F = [[sb[j] / (sb.max() + 1e-9), sl[j], len(qt & titulos[j]) / max(1, len(qt)), len(qt & vocab[j]) / max(1, len(qt))] for j in top]
    return top, np.array(F), float(sb.max())

info = [candidatos(c[0]) for c in CONSULTAS]
tem = np.array([bool(c[1]) for c in CONSULTAS])
rng = np.random.default_rng(0)
metade = np.zeros(len(CONSULTAS), int)
for grupo in (np.where(tem)[0], np.where(~tem)[0]):
    p = rng.permutation(grupo); metade[p[::2]] = 0; metade[p[1::2]] = 1

linhas = {"decisao": [], "recusa_sem": [], "resp_com": [], "mrr_rrf": [], "mrr_rerank": []}
for ajuste_id in (0, 1):
    aj = np.where(metade == ajuste_id)[0]; te = np.where(metade != ajuste_id)[0]
    aj_com = [i for i in aj if tem[i]]
    X = np.vstack([info[i][1] for i in aj_com]); y = np.concatenate([[ids[j] in CONSULTAS[i][1] for j in info[i][0]] for i in aj_com]).astype(int)
    modelo = LogisticRegression(max_iter=1000).fit(X, y)
    grade = np.arange(0, 12, 0.1)
    lim = max(grade, key=lambda l: (np.mean([(info[i][2] > l) == tem[i] for i in aj]), -l))
    for i in te:
        aceita = info[i][2] > lim
        linhas["decisao"].append(float(aceita == tem[i]))
        if not tem[i]:
            linhas["recusa_sem"].append(float(not aceita))
        else:
            linhas["resp_com"].append(float(aceita))
            if aceita:
                top, F, _ = info[i]; rel = CONSULTAS[i][1]
                linhas["mrr_rrf"].append(av.mrr([ids[j] for j in top], rel))
                ordem = np.argsort(-modelo.decision_function(F), kind="stable")
                linhas["mrr_rerank"].append(av.mrr([ids[top[j]] for j in ordem], rel))
    print(f"ajuste na metade {ajuste_id}: limiar escolhido = {lim:.1f}")

nomes = {"decisao": "acerto da decisão (56)", "recusa_sem": "recusa das sem resposta", "resp_com": "resposta das com resposta",
         "mrr_rrf": "MRR aceitas, RRF", "mrr_rerank": "MRR aceitas, reranqueado"}
print(f"{'métrica':<28}{'n':>4}{'valor':>8}{'IC95%':>20}")
for k, nome in nomes.items():
    m, lo, hi = av.bootstrap_ic(linhas[k], n=2000, seed=0)
    print(f"{nome:<28}{len(linhas[k]):>4}{m:>8.3f}{f'[{lo:.3f}; {hi:.3f}]':>20}")

