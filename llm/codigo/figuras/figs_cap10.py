"""Figuras do Capítulo 10. As marcadas [D] são calculadas com a base de 60 documentos ou com a coleção sintética."""
import sys; sys.path.insert(0, "/home/user/claude/llm/build"); sys.path.insert(0, "/home/user/claude/llm/codigo/comum")
from fc import *
from llmdidatico import DOCS, CONSULTAS, modelos, avaliacao as av
from llmdidatico.indices import rrf, chunk_palavras, colecao_sintetica, IndiceIVF, IndiceLSH

ids = [d[0] for d in DOCS]; textos = [d[2] + ". " + d[3] for d in DOCS]; tok = modelos.tokenizar_sem_stop
bm = modelos.BM25([tok(t) for t in textos]); lsa = modelos.LSAEmbedder(dim=40).ajustar(textos); Dl = lsa.transformar(textos)
COM = [c for c in CONSULTAS if c[1]]; SEM = [c for c in CONSULTAS if not c[1]]
rk_bm = lambda q: [ids[i] for i in np.argsort(-bm.pontuar(tok(q)), kind="stable")]
rk_ls = lambda q: [ids[i] for i in np.argsort(-(Dl @ lsa.transformar([q])[0]), kind="stable")]
rk_rrf = lambda q: rrf([rk_bm(q), rk_ls(q)])[0]


def mapa():
    fig, ax = canvas(9.4, 4.4)
    rbox(ax, 37, 78, 26, 16, "Busca semântica\ne recuperação", fc=IND, ec=IND, tc="white", fs=10, bold=True)
    itens = [(2, 52, "Representar\ntexto → vetor\nTF-IDF · LSA · densos", CYA), (35, 52, "Comparar\ncosseno · produto\neuclidiana", GRN),
             (68, 52, "Recuperar\nBM25 · densa · híbrida\nRRF", AMB), (2, 22, "Preparar\nsegmentação (chunking)\nmetadados · filtros", VIO),
             (35, 22, "Acelerar\nbusca exata · IVF · LSH\nmemória N·d·bytes", PNK), (68, 22, "Refinar\nreranqueamento\nlimiar · sem resposta", RED)]
    for x, y, t, c in itens:
        rbox(ax, x, y, 30, 20, t, fc="white", ec=c, fs=8.4, bold=True, lw=1.8); (arrow(ax, (50, 77.5), (x + 15, y + 20.5), c=GRY, lw=1.1) if y > 40 else arrow(ax, (x + 15, 51.5), (x + 15, y + 20.5), c=GRY, lw=1.1))
    rbox(ax, 20, 2, 60, 10, "Avaliar: recall@k · hit · MRR · nDCG · IC por bootstrap", fc="#E0E7FF", ec=IND, fs=9, bold=True)
    save(fig, "f10_mapa")


def pipeline():
    fig, ax = canvas(9.4, 4.0)
    ax.text(2, 96, "Indexação (offline)", fontsize=9.5, fontweight="bold", color=IND); ax.text(2, 47, "Consulta (online)", fontsize=9.5, fontweight="bold", color=GRN)
    off = ["Documentos", "Segmentar\n(chunks)", "Embedding\n+ metadados", "Índice\nvetorial"]
    for k, t in enumerate(off):
        rbox(ax, 2 + k * 25, 72, 21, 18, t, fc="white", ec=IND, fs=8.6, bold=True)
        if k: arrow(ax, (2 + k * 25 - 3.6, 81), (2 + k * 25 - 0.2, 81), c=INK)
    on = ["Consulta", "Embedding\n(mesmo modelo)", "Busca no índice\n+ filtros (top-N)", "Reranqueador\n(top-k)", "Contexto\npara o LLM"]
    for k, t in enumerate(on):
        rbox(ax, 1 + k * 20, 20, 17, 18, t, fc="white", ec=GRN, fs=8, bold=True)
        if k: arrow(ax, (1 + k * 20 - 2.8, 29), (1 + k * 20 - 0.2, 29), c=INK)
    arrow(ax, (87, 71.5), (53, 38.5), c=GRY, ls="--", rad=-0.15)
    save(fig, "f10_pipeline")


def similaridade():
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    q = np.array([1, 0.0]); pts_ = {"A": np.array([3.0, 3.0]), "B": np.array([1.0, 0.1]), "C": np.array([0.5, 0.0])}
    ax.annotate("", xy=q, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=2)); ax.text(1.05, -0.28, "consulta q", fontsize=8.5, color=INK)
    for (n, v), c in zip(pts_.items(), [RED, GRN, AMB]):
        ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.6)); ax.text(v[0] + 0.07, v[1] + 0.1, n, color=c, fontweight="bold")
    ax.text(0.1, 2.85, "produto interno: A > B > C\ncosseno: C = 1 > B > A\ndistância: B < C < A", fontsize=8.5, va="top", bbox=dict(fc=LIGHT, ec=MID, boxstyle="round"))
    ax.set_xlim(-0.3, 3.8); ax.set_ylim(-0.5, 3.6); ax.set_aspect("equal"); ax.axhline(0, c=MID, lw=0.6); ax.axvline(0, c=MID, lw=0.6)
    ax.set_title("Três medidas, três ordens (vetores não normalizados)", fontsize=9.5); save(fig, "f10_similaridade")


def metodos():
    nomes = ["BM25", "LSA", "RRF (k=60)", "Soma\nnormalizada"]
    def soma(q):
        sb = bm.pontuar(tok(q)); sl = Dl @ lsa.transformar([q])[0]; nz = lambda s: (s - s.min()) / (s.max() - s.min() + 1e-12)
        return [ids[i] for i in np.argsort(-(0.5 * nz(sb) + 0.5 * nz(sl)), kind="stable")]
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.4), sharey=True)
    for ax, (mn, f) in zip(axs, [("MRR", lambda rk, rel: av.mrr(rk, rel)), ("hit@3", lambda rk, rel: av.hit_at_k(rk, rel, 3))]):
        for k, (n, r) in enumerate(zip(nomes, [rk_bm, rk_ls, rk_rrf, soma])):
            v = np.array([f(r(c[0]), c[1]) for c in COM]); m, lo, hi = av.bootstrap_ic(v, n=2000, seed=0)
            ax.errorbar(k, m, yerr=[[m - lo], [hi - m]], fmt="o", c=CL[k], capsize=4, ms=7)
            ax.text(k, hi + 0.012, f"{m:.2f}", ha="center", fontsize=8)
        ax.set_xticks(range(4)); ax.set_xticklabels(nomes, fontsize=8); ax.set_title(f"{mn} (IC 95%, 46 consultas)", fontsize=9); ax.set_ylim(0.7, 1.02)
    axs[0].set_ylabel("valor da métrica"); fig.tight_layout(); save(fig, "f10_metodos")


def rrf_curva():
    r = np.arange(1, 31); fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for k, c in zip((1, 10, 60), (RED, AMB, IND)): ax.plot(r, 1 / (k + r) / (1 / (k + 1)), "o-", ms=3, c=c, label=f"k = {k}")
    ax.set_xlabel("posição no ranking (r)"); ax.set_ylabel("contribuição 1/(k+r), relativa à posição 1"); ax.legend(frameon=False)
    ax.set_title("RRF: k pequeno privilegia o topo; k grande suaviza", fontsize=9.5); save(fig, "f10_rrf")


def chunks():
    manual = " ".join(textos); inicio = []; pos = 0
    for t in textos: n = len(t.split()); inicio.append((pos, pos + n)); pos += n
    res = {}
    for q, rel, ans, _ in COM:
        i = ids.index(rel[0]); k = textos[i].find(ans); w0 = inicio[i][0] + len(textos[i][:k].split()); res[q] = (w0, w0 + len(ans.split()))
    def aval(ped):
        b = modelos.BM25([tok(p[2]) for p in ped]); h, w = [], []
        for q, (a0, a1) in res.items():
            top = np.argsort(-b.pontuar(tok(q)), kind="stable")[:3]
            h.append(any(ped[j][0] <= a0 and ped[j][1] >= a1 for j in top)); w.append(sum(ped[j][1] - ped[j][0] for j in top))
        return np.mean(h), np.mean(w)
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    for sob, c, lab in ((0, IND, "sem sobreposição"), (0.5, GRN, "sobreposição de 50%")):
        xs, ys = [], []
        for tam in (10, 20, 40, 80, 160):
            h, w = aval(chunk_palavras(manual, tam, int(tam * sob))); xs.append(w); ys.append(h)
            ax.text(w, h + (0.03 if sob else -0.05), f"{tam}", fontsize=7.5, ha="center", color=c)
        ax.plot(xs, ys, "o-", c=c, label=lab)
    h, w = aval([(a, b, t) for (a, b), t in zip(inicio, textos)]); ax.plot([w], [h], "s", c=RED, ms=8, label="1 documento = 1 pedaço")
    ax.set_xlabel("palavras enviadas ao modelo (3 melhores pedaços)"); ax.set_ylabel("hit@3 (resposta inteira no contexto)"); ax.set_ylim(0.3, 1.02)
    ax.legend(frameon=False, fontsize=8, loc="lower right"); ax.set_title("Segmentação: tamanho do pedaço (rótulos) e custo", fontsize=9.5); save(fig, "f10_chunks")


def ann():
    D, Q = colecao_sintetica(N=20000, d=64, seed=0); verdade = np.argsort(-(Q @ D.T), axis=1)[:, :10]
    def rf(b):
        r, f = [], []
        for i, q in enumerate(Q): t, n = b(q); r.append(len(set(t) & set(verdade[i])) / 10); f.append(n / len(D))
        return np.mean(r), np.mean(f)
    ivf = IndiceIVF(100).treinar(D); P = [rf(lambda q, n=n: ivf.buscar(q, 10, n)) for n in (1, 2, 4, 8, 16, 32)]
    L = []
    for nt in (2, 4, 8, 16, 32, 64): l = IndiceLSH(nt, 8).treinar(D); L.append(rf(lambda q: l.buscar(q, 10)))
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.plot([p[1] for p in P], [p[0] for p in P], "o-", c=IND, label="IVF (k-means, 100 listas)"); ax.plot([p[1] for p in L], [p[0] for p in L], "s-", c=AMB, label="LSH (8 bits, 2 a 64 tabelas)")
    ax.plot([0, 1], [0, 1], "--", c=GRY, lw=1, label="examinar ao acaso"); ax.plot([1], [1], "*", c=RED, ms=12, label="busca exaustiva")
    ax.set_xlabel("fração da coleção examinada"); ax.set_ylabel("recall@10 contra a busca exata"); ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8, loc="lower right"); ax.set_title("Busca aproximada: recall × esforço (coleção sintética, N = 20 000)", fontsize=9); save(fig, "f10_ann")


def limiar():
    f = lambda q: float(bm.pontuar(tok(q)).max()); A = [f(c[0]) for c in COM]; U = [f(c[0]) for c in SEM]
    fig, ax = plt.subplots(figsize=(5.8, 3.4)); b = np.arange(0, 20, 1.0)
    ax.hist(A, bins=b, alpha=0.7, color=IND, label=f"com resposta (n={len(A)})"); ax.hist(U, bins=b, alpha=0.7, color=RED, label=f"sem resposta (n={len(U)})")
    ax.axvline(3.4, c=INK, ls="--", lw=1.2); ax.text(3.6, ax.get_ylim()[1] * 0.9, "limiar 3,4", fontsize=8)
    ax.set_xlabel("pontuação BM25 do 1º colocado"); ax.set_ylabel("número de consultas"); ax.legend(frameon=False, fontsize=8); ax.set_title("As duas populações se sobrepõem", fontsize=9.5); save(fig, "f10_limiar")


def bicross():
    fig, ax = canvas(9.4, 3.8)
    ax.text(2, 96, "Bi-encoder (rápido; vetores pré-calculados)", fontsize=9.5, fontweight="bold", color=IND); ax.text(54, 96, "Cross-encoder (lento; olha o par junto)", fontsize=9.5, fontweight="bold", color=RED)
    for k, t in enumerate(["consulta", "documento"]):
        y = 64 - k * 30; rbox(ax, 1, y, 13, 14, t, fc=LIGHT, ec=GRY, fs=8.2); rbox(ax, 18, y, 14, 14, "codificador", fc="white", ec=IND, fs=8, bold=True)
        rbox(ax, 35, y, 6, 14, "v", fc="#E0E7FF", ec=IND, fs=9, bold=True); arrow(ax, (14.2, y + 7), (17.8, y + 7)); arrow(ax, (32.2, y + 7), (34.8, y + 7))
    rbox(ax, 43, 46, 9, 12, "cos\n(vq, vd)", fc="#FEF3C7", ec=AMB, fs=8, bold=True)
    arrow(ax, (41.2, 70), (47.5, 58.3), c=GRY); arrow(ax, (41.2, 40), (47.5, 45.7), c=GRY)
    rbox(ax, 56, 60, 18, 16, "consulta + documento\n(juntos)", fc=LIGHT, ec=GRY, fs=7.8); rbox(ax, 78, 56, 21, 24, "codificador\ncom atenção\nentre os dois", fc="white", ec=RED, fs=8.2, bold=True)
    arrow(ax, (74.2, 68), (77.8, 68)); rbox(ax, 78, 22, 21, 12, "pontuação de\nrelevância", fc="#FEE2E2", ec=RED, fs=8.2, bold=True); arrow(ax, (88.5, 55.5), (88.5, 34.5))
    ax.text(2, 8, "Uso típico: o bi-encoder recupera 100 candidatos; o cross-encoder reordena os 10 melhores.", fontsize=8.5, color=INK)
    save(fig, "f10_bicross")


def memoria():
    fig, ax = plt.subplots(figsize=(5.8, 3.4)); cfg = [("1e5 × 384", 1e5, 384), ("1e6 × 768", 1e6, 768), ("1e7 × 768", 1e7, 768), ("1e8 × 1536", 1e8, 1536)]
    x = np.arange(len(cfg)); w = 0.26
    for k, (b, n, c) in enumerate([(4, "float32", IND), (2, "float16", CYA), (1, "int8", GRN)]): ax.bar(x + (k - 1) * w, [N * d * b / 1e9 for _, N, d in cfg], w, color=c, label=n)
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([c[0] for c in cfg], fontsize=8.5); ax.set_ylabel("memória dos vetores (GB, escala log)"); ax.set_xlabel("N documentos × d dimensões")
    ax.legend(frameon=False, fontsize=8); ax.set_title("Memória = N · d · bytes por número", fontsize=9.5); save(fig, "f10_memoria")


if __name__ == "__main__":
    mapa(); pipeline(); similaridade(); metodos(); rrf_curva(); chunks(); ann(); limiar(); bicross(); memoria()
