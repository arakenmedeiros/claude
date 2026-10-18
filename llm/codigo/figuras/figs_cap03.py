"""Figuras do Capítulo 3. f03_fertilidade e f03_embeddings são baseados em dados; f03_mapa e f03_pipeline são esquemas."""
import sys; sys.path.insert(0, "/home/user/claude/llm/build"); sys.path.insert(0, "/home/user/claude/llm/codigo/comum")
import re
from fc import *
from llmdidatico import DOCS, corpus, modelos


def mapa():
    fig, ax = canvas(9.4, 3.6)
    etapas = [("Texto\n\"A matrícula\nfoi trancada.\"", "#E0E7FF", IND), ("Normalização\nUnicode, caixa,\nacentos (opcional)", "#CFFAFE", CYA), ("Tokenização\nsubpalavras\n(BPE, WordPiece…)", "#D1FAE5", GRN),
              ("Identificadores\n[12, 3871, 40,\n9120, 5]", "#FEF3C7", AMB), ("Embeddings\nmatriz T × d", "#FCE7F3", PNK)]
    for k, (t, fc_, ec) in enumerate(etapas):
        x = 1 + k * 19.6
        rbox(ax, x, 38, 18, 40, t, fc=fc_, ec=ec, fs=7.6, bold=True)
        if k < 4: arrow(ax, (x + 17.4, 58), (x + 19.2, 58), c=INK)
    rbox(ax, 62, 6, 36, 20, "Busca semântica:\ncosseno entre vetores", fc="#EEF2FF", ec=IND, fs=9, bold=True)
    arrow(ax, (88, 37.5), (80, 26.5), c=INK)
    rbox(ax, 2, 6, 54, 20, "Cada etapa pode distorcer: normalização perde acentos, tokenização fragmenta,\nembeddings herdam vieses dos dados.", fc="#FEE2E2", ec=RED, fs=7.4)
    save(fig, "f03_mapa")


def pipeline():
    fig, ax = canvas(9.2, 3.3)
    toks = ["A", "matrí", "cula", "foi", "tranc", "ada", "."]; ids = [12, 3871, 951, 40, 8817, 302, 5]
    cols = [IND, CYA, CYA, GRN, AMB, AMB, GRY]
    for k, (t, i, c) in enumerate(zip(toks, ids, cols)):
        x = 3 + k * 13.4
        rbox(ax, x, 62, 12, 14, t, fc=c, ec=c, tc="white", fs=10, bold=True)
        ax.text(x + 6, 52, str(i), ha="center", fontsize=9, color=INK)
        arrow(ax, (x + 6, 60.5), (x + 6, 55), c=GRY, lw=1.0, ms=8)
        rbox(ax, x, 12, 12, 30, "", fc="white", ec=c, lw=1.6)
        rng = np.random.default_rng(i)
        for r in range(6): ax.add_patch(Rectangle((x + 1.2, 14 + r * 4.4), 9.6, 3.4, fc=c, alpha=float(0.25 + 0.7 * rng.random()), ec="none"))
        arrow(ax, (x + 6, 50), (x + 6, 43), c=GRY, lw=1.0, ms=8)
    ax.text(1, 90, "texto → tokens (subpalavras) → identificadores → linhas da matriz de embeddings", fontsize=10, fontweight="bold", color=INK)
    ax.text(1, 3, "Os identificadores e os pedaços são ilustrativos; cada modelo tem o seu vocabulário.", fontsize=8, color=GRY, style="italic")
    save(fig, "f03_pipeline")


def fertilidade():
    texto = " ".join(d[2] + " " + d[3] for d in DOCS).lower()
    palavras = re.findall(r"[a-zçáàâãéêíóôõú]+", texto)
    ns = [0, 25, 50, 100, 150, 200, 300, 400, 600]
    vals = []
    for n in ns:
        f = modelos.treinar_bpe(palavras, n); vals.append(np.mean([len(modelos.codificar_bpe(w, f)) for w in palavras]))
    fig, ax = plt.subplots(figsize=(6.2, 3.6)); ax.plot(ns, vals, "-o", c=IND, lw=2.2)
    for n, v in zip(ns, vals):
        if n in (0, 100, 400): ax.text(n + 12, v + 0.12, f"{v:.2f}", fontsize=9)
    ax.set_xlabel("número de fusões (acréscimos ao vocabulário)"); ax.set_ylabel("tokens por palavra (média)"); ax.set_title("Mais fusões, sequências mais curtas"); ax.set_ylim(0, 6.6)
    save(fig, "f03_fertilidade")


def embeddings():
    frases = corpus.frases_lm(3000, seed=0); vocab = sorted({t for f in frases for t in f}); ix = {t: i for i, t in enumerate(vocab)}; V = len(vocab)
    C = np.zeros((V, V))
    for f in frases:
        for i, t in enumerate(f):
            for j in range(max(0, i - 2), min(len(f), i + 3)):
                if j != i: C[ix[t], ix[f[j]]] += 1
    tot = C.sum(); pw = C.sum(1, keepdims=True) / tot; pc = C.sum(0, keepdims=True) / tot
    with np.errstate(divide="ignore", invalid="ignore"): pmi = np.log((C / tot) / (pw * pc))
    ppmi = np.where(np.isfinite(pmi), np.maximum(pmi, 0), 0.0)
    U, S, _ = np.linalg.svd(ppmi, full_matrices=False); E = U[:, :8] * np.sqrt(S[:8])
    Ec = E - E.mean(0); u, s, vt = np.linalg.svd(Ec, full_matrices=False); P = Ec @ vt[:2].T; var = (s ** 2 / (s ** 2).sum())[:2].sum()
    grupos = {"sujeitos": ["estudante", "professor", "professora", "coordenadora", "bibliotecário", "aluno"], "verbos": ["lê", "escreve", "entrega", "empresta", "corrige", "explica", "devolve", "estuda", "procura", "assina", "organiza", "prepara"],
              "objetos": ["livro", "artigo", "relatório", "prova", "trabalho", "aula"], "tempo/lugar": ["hoje", "ontem", "casa", "biblioteca", "laboratório"]}
    cor = {"sujeitos": IND, "verbos": RED, "objetos": GRN, "tempo/lugar": AMB}
    fig, ax = plt.subplots(figsize=(7.4, 5))
    for g, ws in grupos.items():
        xs = [P[ix[w], 0] for w in ws if w in ix]; ys = [P[ix[w], 1] for w in ws if w in ix]; ax.scatter(xs, ys, c=cor[g], s=40, label=g, zorder=3)
        for w in ws:
            if w in ix: ax.text(P[ix[w], 0], P[ix[w], 1], " " + w, fontsize=7.5, color=cor[g])
    ax.legend(fontsize=8); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"Embeddings de palavras (PCA 2D; {var:.0%} da variância dos vetores de 8 dim.)")
    save(fig, "f03_embeddings")


if __name__ == "__main__":
    mapa(); pipeline(); fertilidade(); embeddings()
