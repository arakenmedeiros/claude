"""Capítulo 10: blocos de código do texto, em ordem. Cada "# %%" é uma célula."""

# %% [bloco 1]
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from llmdidatico import DOCS, CONSULTAS, modelos, avaliacao as av
from llmdidatico.indices import normalizar_l2, pontuar, knn_exato, rrf, chunk_palavras

ids = [d[0] for d in DOCS]
areas = [d[1] for d in DOCS]
textos = [d[2] + ". " + d[3] for d in DOCS]
tok = modelos.tokenizar_sem_stop

cont = CountVectorizer(tokenizer=tok, lowercase=False, token_pattern=None)
Xc = cont.fit_transform(textos)                       # contagens (esparsa)
tfidf = TfidfVectorizer(tokenizer=tok, lowercase=False, token_pattern=None, sublinear_tf=True)
Xt = tfidf.fit_transform(textos)                      # TF-IDF (esparsa)
lsa = modelos.LSAEmbedder(dim=40).ajustar(textos)
Dl = lsa.transformar(textos)                          # embeddings densos (60 x 40), norma 1

print("contagens:", Xc.shape, f"| fração de zeros: {1 - Xc.nnz / (Xc.shape[0] * Xc.shape[1]):.1%}")
print("LSA      :", Dl.shape, "| normas:", np.linalg.norm(Dl, axis=1).round(3)[:3], "...")
for alvo in ("D12", "D37"):
    i = ids.index(alvo)
    s = Dl @ Dl[i]
    viz = [(ids[j], round(float(s[j]), 2)) for j in np.argsort(-s)[1:4]]
    print(f"vizinhos de {alvo} ({DOCS[i][2]}):", viz)


# %% [bloco 2]
# vetores de exemplo (2D)
q = np.array([1.0, 0.0])
cand = {"A": np.array([3.0, 3.0]), "B": np.array([1.0, 0.1]), "C": np.array([0.5, 0.0])}
print(f"{'doc':<4}{'produto':>9}{'cosseno':>9}{'dist.euclid.':>14}")
for nome, v in cand.items():
    print(f"{nome:<4}{q @ v:>9.3f}{q @ v / (np.linalg.norm(q) * np.linalg.norm(v)):>9.3f}{np.linalg.norm(q - v):>14.3f}")

# com vetores unitários: ||a-b||^2 = 2 - 2 cos(a, b)
a, b = Dl[ids.index("D12")], Dl[ids.index("D11")]
print("||a-b||^2 =", round(float(np.sum((a - b) ** 2)), 6), "| 2 - 2cos =", round(float(2 - 2 * a @ b), 6))

# em vetores NÃO normalizados (contagens), o produto interno favorece textos longos
longo = " ".join(textos[:6])                           # um documento artificial, longo
colecao = textos + [longo]
X = CountVectorizer(tokenizer=tok, lowercase=False, token_pattern=None).fit(textos)
M = X.transform(colecao).toarray().astype(float)
qv = X.transform(["Quanto tempo posso trancar a matrícula e quando posso cancelar disciplina?"]).toarray().astype(float)
for metrica in ("produto", "cosseno", "euclidiana"):
    ordem = np.argsort(-pontuar(qv, M, metrica)[0], kind="stable")[:3]
    print(f"{metrica:<11} top-3:", ["LONGO" if j == 60 else ids[j] for j in ordem])


# %% [bloco 3]
ranking = ["D05", "D12", "D03", "D40", "D11"]
relevantes = {"D12", "D11"}
print("recall@3 =", av.recall_at_k(ranking, relevantes, 3))
print("hit@1    =", av.hit_at_k(ranking, relevantes, 1), "| hit@3 =", av.hit_at_k(ranking, relevantes, 3))
print("MRR      =", av.mrr(ranking, relevantes))
dcg = 1 / np.log2(3) + 1 / np.log2(6)
idcg = 1 / np.log2(2) + 1 / np.log2(3)
print(f"DCG@5 = {dcg:.4f} | IDCG@5 = {idcg:.4f} | nDCG@5 = {dcg / idcg:.4f} | função: {av.ndcg_at_k(ranking, relevantes, 5):.4f}")

COM_RESP = [c for c in CONSULTAS if c[1]]             # 46 consultas com resposta na base
SEM_RESP = [c for c in CONSULTAS if not c[1]]         # 10 sem resposta
print(len(COM_RESP), "com resposta |", len(SEM_RESP), "sem resposta")

def metricas(ranker, consultas=COM_RESP):
    """Vetores por consulta: hit@1, hit@3, recall@3, MRR, nDCG@5."""
    saida = {"hit@1": [], "hit@3": [], "recall@3": [], "MRR": [], "nDCG@5": []}
    for q, rel, _, _ in consultas:
        rk = ranker(q)
        saida["hit@1"].append(av.hit_at_k(rk, rel, 1)); saida["hit@3"].append(av.hit_at_k(rk, rel, 3))
        saida["recall@3"].append(av.recall_at_k(rk, rel, 3)); saida["MRR"].append(av.mrr(rk, rel))
        saida["nDCG@5"].append(av.ndcg_at_k(rk, rel, 5))
    return {k: np.array(v) for k, v in saida.items()}

def resumo(nome, m):
    cols = []
    for k, v in m.items():
        mu, lo, hi = av.bootstrap_ic(v, n=2000, seed=0)
        cols.append(f"{mu:.3f} [{lo:.3f};{hi:.3f}]")
    print(f"{nome:<8}" + "  ".join(cols))


# %% [bloco 4]
bm = modelos.BM25([tok(t) for t in textos])

def rank_bm25(q):
    return [ids[i] for i in np.argsort(-bm.pontuar(tok(q)), kind="stable")]

def rank_lsa(q):
    return [ids[i] for i in np.argsort(-(Dl @ lsa.transformar([q])[0]), kind="stable")]

print(f"{'método':<8}" + "  ".join(f"{k:^19}" for k in ["hit@1", "hit@3", "recall@3", "MRR", "nDCG@5"]))
M_BM, M_LSA = metricas(rank_bm25), metricas(rank_lsa)
resumo("BM25", M_BM); resumo("LSA", M_LSA)

tipos = np.array([c[3] for c in COM_RESP])
for tipo in ("direta", "parafrase"):
    for nome, m in (("BM25", M_BM), ("LSA", M_LSA)):
        print(f"{tipo:<10}{nome:<6} n={int((tipos == tipo).sum()):<3} MRR = {m['MRR'][tipos == tipo].mean():.3f}   hit@3 = {m['hit@3'][tipos == tipo].mean():.3f}")

def dif_pareada(a, b, n=5000, seed=0):
    """IC bootstrap da diferença média (b - a), reamostrando as MESMAS consultas nos dois métodos."""
    rng = np.random.default_rng(seed); d = np.asarray(b) - np.asarray(a)
    medias = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return d.mean(), np.quantile(medias, 0.025), np.quantile(medias, 0.975)

m, lo, hi = dif_pareada(M_BM["MRR"], M_LSA["MRR"])
print(f"MRR(LSA) - MRR(BM25) = {m:+.3f}  IC95% [{lo:+.3f}; {hi:+.3f}]")


# %% [bloco 5]
import importlib.util
import transformers

print("sentence_transformers instalado neste ambiente?", importlib.util.find_spec("sentence_transformers") is not None)
print("transformers", transformers.__version__, "->", {n: hasattr(transformers, n) for n in ["AutoTokenizer", "AutoModel", "AutoModelForSequenceClassification"]})


# %% [bloco 6]
# exemplo de RRF feito à mão (k = 60)
A = ["D1", "D2", "D3", "D4"]            # ranking do método A
B = ["D3", "D1", "D5", "D2"]            # ranking do método B
fundido, pont = rrf([A, B], k=60)
for d in fundido:
    print(d, round(pont[d], 5))
print("à mão: D1 =", round(1 / 61 + 1 / 62, 5), "| D3 =", round(1 / 63 + 1 / 61, 5), "| D4 =", round(1 / 64, 5))

def rank_rrf(q, k=60):
    return rrf([rank_bm25(q), rank_lsa(q)], k=k)[0]

def rank_soma(q, alpha=0.5):
    """Fusão por pontuação: normaliza cada pontuação (min-max) e combina linearmente."""
    sb = bm.pontuar(tok(q)); sl = Dl @ lsa.transformar([q])[0]
    nz = lambda s: (s - s.min()) / (s.max() - s.min() + 1e-12)
    return [ids[i] for i in np.argsort(-(alpha * nz(sb) + (1 - alpha) * nz(sl)), kind="stable")]

print(f"{'método':<14}" + "  ".join(f"{k:^19}" for k in ["hit@1", "hit@3", "recall@3", "MRR", "nDCG@5"]))
resumo("BM25", M_BM); resumo("LSA", M_LSA)
for k in (1, 10, 60):
    resumo(f"RRF k={k}", metricas(lambda q, k=k: rank_rrf(q, k)))
resumo("soma 0,5", metricas(rank_soma))
M_RRF = metricas(rank_rrf)
m, lo, hi = dif_pareada(M_BM["MRR"], M_RRF["MRR"])
print(f"MRR(RRF) - MRR(BM25) = {m:+.3f}  IC95% [{lo:+.3f}; {hi:+.3f}]")


# %% [bloco 7]
# um "manual" único com os 60 documentos em sequência, e a posição (em palavras) de cada documento
inicio, pos = [], 0
for t in textos:
    n = len(t.split()); inicio.append((pos, pos + n)); pos += n
manual = " ".join(textos)
print("manual:", pos, "palavras")

# onde está a resposta de cada consulta (intervalo de palavras no manual)
resposta_em = {}
for q, rel, ans, _ in COM_RESP:
    i = ids.index(rel[0]); k = textos[i].find(ans)
    if k >= 0:
        w0 = inicio[i][0] + len(textos[i][:k].split()); resposta_em[q] = (w0, w0 + len(ans.split()))
print("consultas com resposta localizada:", len(resposta_em), "de", len(COM_RESP))

def avalia_chunks(pedacos, k=3):
    """hit@k: algum dos k melhores pedaços contém INTEIRA a passagem da resposta? Também mede o contexto enviado."""
    bmc = modelos.BM25([tok(p[2]) for p in pedacos]); acertos, palavras = [], []
    for q, (a0, a1) in resposta_em.items():
        topo = np.argsort(-bmc.pontuar(tok(q)), kind="stable")[:k]
        acertos.append(any(pedacos[j][0] <= a0 and pedacos[j][1] >= a1 for j in topo))
        palavras.append(sum(pedacos[j][1] - pedacos[j][0] for j in topo))
    return np.mean(acertos), np.mean(palavras)

print(f"{'estratégia':<26}{'pedaços':>8}{'hit@3':>8}{'palavras no contexto':>22}")
for tam in (10, 20, 40, 80, 160):
    for sob in (0, tam // 2):
        p = chunk_palavras(manual, tam, sob); h, w = avalia_chunks(p)
        print(f"{f'{tam} palavras, sobrep. {sob}':<26}{len(p):>8}{h:>8.3f}{w:>22.0f}")
por_doc = [(a, b, t) for (a, b), t in zip(inicio, textos)]
h, w = avalia_chunks(por_doc)
print(f"{'por estrutura (1 doc)':<26}{len(por_doc):>8}{h:>8.3f}{w:>22.0f}")


# %% [bloco 8]
from llmdidatico.indices import colecao_sintetica, IndiceIVF, IndiceLSH

D, Q = colecao_sintetica(N=20000, d=64, seed=0)       # SINTÉTICO: 20 000 vetores unitários, 200 consultas
verdade = np.argsort(-(Q @ D.T), axis=1)[:, :10]      # busca exaustiva = padrão-ouro
print("busca exaustiva:", D.shape, "->", f"{D.astype(np.float32).nbytes / 1e6:.1f} MB em float32")

def recall_e_fracao(buscar):
    r, f = [], []
    for i, q in enumerate(Q):
        top, n = buscar(q)
        r.append(len(set(top) & set(verdade[i])) / 10); f.append(n / len(D))
    return np.mean(r), np.mean(f)

ivf = IndiceIVF(n_listas=100).treinar(D)
print(f"{'IVF (100 listas)':<22}{'recall@10':>10}{'fração examinada':>18}")
for nprobe in (1, 2, 4, 8, 16, 32):
    r, f = recall_e_fracao(lambda q: ivf.buscar(q, 10, nprobe))
    print(f"{f'nprobe = {nprobe}':<22}{r:>10.3f}{f:>18.3f}")
print(f"{'LSH (hiperplanos)':<22}{'recall@10':>10}{'fração examinada':>18}")
for nt, nb in [(4, 8), (8, 8), (16, 8), (32, 8), (16, 12), (32, 10)]:
    lsh = IndiceLSH(nt, nb).treinar(D)
    r, f = recall_e_fracao(lambda q: lsh.buscar(q, 10))
    print(f"{f'{nt} tabelas, {nb} bits':<22}{r:>10.3f}{f:>18.3f}")

# memória: N x d x bytes
print(f"{'N':>12}{'d':>6}{'float32':>10}{'float16':>10}{'int8':>10}")
for N, d in [(1e5, 384), (1e6, 768), (1e7, 768), (1e8, 1536)]:
    print(f"{int(N):>12,}{d:>6}" + "".join(f"{N * d * b / 1e9:>9.2f}G" for b in (4, 2, 1)))

# quantização escalar para int8: perde pouco?
esc = np.abs(D).max()
D8 = np.round(D / esc * 127).astype(np.int8)
S8 = (Q / esc * 127) @ D8.T.astype(np.float32)
top8 = np.argsort(-S8, axis=1)[:, :10]
print("recall@10 de int8 contra float32:", round(float(np.mean([len(set(top8[i]) & set(verdade[i])) / 10 for i in range(len(Q))])), 3))


# %% [bloco 9]
from sklearn.linear_model import LogisticRegression
titulos = [set(tok(d[2])) for d in DOCS]; vocab_doc = [set(tok(t)) for t in textos]
bigramas = lambda t: set(zip(t, t[1:]))
big_doc = [bigramas(tok(t)) for t in textos]

def candidatos(q, K=10):
    """Primeira etapa: RRF de BM25 e LSA (K candidatos) + características que olham o PAR (consulta, documento)."""
    sb = bm.pontuar(tok(q)); sl = Dl @ lsa.transformar([q])[0]
    rb, rl = list(np.argsort(-sb, kind="stable")), list(np.argsort(-sl, kind="stable"))
    top = [ids.index(d) for d in rrf([[ids[i] for i in rb], [ids[i] for i in rl]])[0][:K]]
    qt, qb = set(tok(q)), bigramas(tok(q)); F = []
    for j in top:
        F.append([sb[j] / (sb.max() + 1e-9), sl[j], len(qt & titulos[j]) / max(1, len(qt)),
                  len(qt & vocab_doc[j]) / max(1, len(qt)), len(qb & big_doc[j]) / max(1, len(qb))])
    return top, np.array(F)

cands = [candidatos(c[0]) for c in COM_RESP]
rotulos = [np.array([ids[j] in c[1] for j in top], int) for (top, _), c in zip(cands, COM_RESP)]
perm = np.random.default_rng(0).permutation(len(COM_RESP)); dobras = np.array_split(perm, 5)
mrr_base, mrr_rerank = np.zeros(len(COM_RESP)), np.zeros(len(COM_RESP))
for teste in dobras:
    treino = [i for i in range(len(COM_RESP)) if i not in teste]
    modelo = LogisticRegression(max_iter=1000).fit(np.vstack([cands[i][1] for i in treino]), np.concatenate([rotulos[i] for i in treino]))
    for i in teste:
        top = cands[i][0]; nova = [ids[top[j]] for j in np.argsort(-modelo.decision_function(cands[i][1]), kind="stable")]
        mrr_base[i] = av.mrr([ids[j] for j in top], COM_RESP[i][1]); mrr_rerank[i] = av.mrr(nova, COM_RESP[i][1])
print("MRR (RRF)        :", round(mrr_base.mean(), 3))
print("MRR (reranqueado):", round(mrr_rerank.mean(), 3))
m, lo, hi = dif_pareada(mrr_base, mrr_rerank)
print(f"diferença pareada: {m:+.3f}  IC95% [{lo:+.3f}; {hi:+.3f}]")
print("consultas que melhoraram / pioraram:", int((mrr_rerank > mrr_base).sum()), "/", int((mrr_rerank < mrr_base).sum()))


# %% [bloco 10]
def melhor_pontuacao(q):
    """Duas pontuações do melhor documento: cosseno LSA e BM25 bruto."""
    return float((Dl @ lsa.transformar([q])[0]).max()), float(bm.pontuar(tok(q)).max())

PA = np.array([melhor_pontuacao(c[0]) for c in COM_RESP]); PU = np.array([melhor_pontuacao(c[0]) for c in SEM_RESP])
print(f"cosseno LSA do 1º colocado: com resposta média {PA[:, 0].mean():.2f} | sem resposta média {PU[:, 0].mean():.2f}")
print(f"BM25 do 1º colocado       : com resposta média {PA[:, 1].mean():.2f} | sem resposta média {PU[:, 1].mean():.2f}")
print(f"{'limiar BM25':>12}{'responde (com resp.)':>22}{'recusa (sem resp.)':>20}")
for lim in (0, 2, 3, 4, 5, 6, 8):
    print(f"{lim:>12}{(PA[:, 1] > lim).mean():>22.2f}{(PU[:, 1] <= lim).mean():>20.2f}")

# escolher o limiar em metade das consultas e medir na outra metade
todas = [(p, 1) for p in PA[:, 1]] + [(p, 0) for p in PU[:, 1]]
rng = np.random.default_rng(1); ordem = rng.permutation(len(todas)); A, B = ordem[::2], ordem[1::2]
def acuracia(lim, idx): return np.mean([(todas[i][0] > lim) == bool(todas[i][1]) for i in idx])
grade = np.arange(0, 10, 0.1)
for nome, ajuste, teste in (("A->B", A, B), ("B->A", B, A)):
    lim = max(grade, key=lambda l: (acuracia(l, ajuste), -l))
    print(f"{nome}: limiar escolhido {lim:.1f} | acerto no ajuste {acuracia(lim, ajuste):.2f} | acerto no teste {acuracia(lim, teste):.2f}")


# %% [bloco 11]
def buscar_com_filtro(q, area=None, k=3, pre=True):
    s = bm.pontuar(tok(q))
    if pre:                                              # pré-filtro: exclui antes de ordenar
        s = np.where([a == area for a in areas], s, -np.inf) if area else s
        return [ids[i] for i in np.argsort(-s, kind="stable")[:k] if np.isfinite(s[i])]
    topo = [ids[i] for i in np.argsort(-s, kind="stable")[:k]]   # pós-filtro: ordena e depois descarta
    return [d for d in topo if not area or areas[ids.index(d)] == area]

q = "Qual o horário de funcionamento e as regras de uso?"
print("sem filtro        :", buscar_com_filtro(q))
print("pós-filtro Campus :", buscar_com_filtro(q, "Campus", pre=False))
print("pré-filtro Campus :", buscar_com_filtro(q, "Campus", pre=True))

# atualização: um documento novo (fictício) com vocabulário que o índice atual não conhece
novo = "Motocicletas: o estacionamento de motocicletas tem 60 vagas gratuitas ao lado da portaria."
q2 = "Onde posso deixar minhas motocicletas?"
ids2 = ids + ["D61"]
def topo(nome, s, rotulos):
    j = int(np.argmax(s)); print(f"{nome:<34} melhor = {rotulos[j]} (pontuação {s[j]:.2f})")
topo("BM25 sem incluir o documento novo", bm.pontuar(tok(q2)), ids)
bm_novo = modelos.BM25([tok(t) for t in textos + [novo]])
topo("BM25 reconstruído com o novo", bm_novo.pontuar(tok(q2)), ids2)
print("palavras do documento novo no vocabulário do LSA antigo:", sum(t in lsa.vec.vocabulary_ for t in tok(novo)), "de", len(tok(novo)))
topo("LSA antigo + vetor do novo", np.vstack([Dl, lsa.transformar([novo])[0]]) @ lsa.transformar([q2])[0], ids2)
lsa2 = modelos.LSAEmbedder(dim=40).ajustar(textos + [novo])
topo("LSA reajustado com o novo", lsa2.transformar(textos + [novo]) @ lsa2.transformar([q2])[0], ids2)

import hashlib
sha = lambda t: hashlib.sha256(t.encode("utf-8")).hexdigest()[:8]
registro = {d: sha(t) for d, t in zip(ids, textos)}
textos_v2 = list(textos); textos_v2[11] = textos[11].replace("R$ 1,50", "R$ 2,00")
mudou = [d for d, t in zip(ids, textos_v2) if registro[d] != sha(t)]
print("documentos a reindexar após a revisão:", mudou)


# %% [bloco 12]
def mostra(q):
    print(f"{q!r}")
    print("   BM25:", rank_bm25(q)[:3], "| LSA:", rank_lsa(q)[:3])

mostra("Posso trancar a matrícula depois da 8ª semana?")
mostra("Não posso trancar a matrícula depois da 8ª semana?")        # negação: quase o mesmo vetor
mostra("Qual benefício paga 150 reais?")                            # o número 150 também é o das vagas do bicicletário
mostra("Qual benefício paga cento e cinquenta reais por mês?")      # número por extenso: sem ligação com '150'
mostra("Como pedir a bolsa de IC?")                                 # sigla ausente do texto do documento
mostra("Como pedir a bolsa de iniciação científica?")
mostra("What is the fine for a late book return?")                  # outro idioma
a, b = lsa.transformar(["Posso trancar a matrícula"])[0], lsa.transformar(["Não posso trancar a matrícula"])[0]
print("cosseno entre afirmação e negação:", round(float(a @ b), 3))
print("tokens de 'R$ 1,50':", tok("R$ 1,50"), "| de 'R$ 150,00':", tok("R$ 150,00"))

