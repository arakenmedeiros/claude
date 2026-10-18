"""Figuras do Capítulo 6. f06_decod e f06_qualidade são baseadas em dados; as demais são esquemas."""
import sys; sys.path.insert(0, "/home/user/claude/llm/build"); sys.path.insert(0, "/home/user/claude/llm/codigo/comum")
from fc import *
from llmdidatico import corpus, modelos


def mapa():
    fig, ax = canvas(9.4, 4.0)
    for k, (t, fc_, ec) in enumerate([("Distribuição do\npróximo token", "#E0E7FF", IND), ("Estratégia de\ndecodificação", "#CFFAFE", CYA), ("Token\nescolhido", "#D1FAE5", GRN), ("Texto gerado,\ntoken a token", "#FEF3C7", AMB)]):
        x = 2 + k * 24.5; rbox(ax, x, 56, 21, 30, t, fc=fc_, ec=ec, fs=9, bold=True)
        if k < 3: arrow(ax, (x + 21.4, 71), (x + 24.1, 71), c=INK)
    rbox(ax, 2, 8, 30, 30, "Janela de contexto\nentrada + saída", fc="#FCE7F3", ec=PNK, fs=9, bold=True)
    rbox(ax, 36, 8, 30, 30, "Memória\ncache KV por sequência", fc="#FEE2E2", ec=RED, fs=9, bold=True)
    rbox(ax, 70, 8, 28, 30, "Latência\nprefill + decode", fc="#EEF2FF", ec=IND, fs=9, bold=True)
    arrow(ax, (50, 55.5), (50, 39), c=GRY, ls="--"); arrow(ax, (85, 55.5), (84, 39), c=GRY, ls="--"); arrow(ax, (17, 55.5), (17, 39), c=GRY, ls="--")
    save(fig, "f06_mapa")


def loop():
    fig, ax = canvas(9.0, 3.8)
    caixas = [("contexto\n\"o aluno\"", IND), ("modelo", CYA), ("distribuição\nprob. do\npróximo token", GRN), ("escolha\n(estratégia)", AMB), ("token\n\"lê\"", PNK)]
    for k, (t, c) in enumerate(caixas):
        x = 2 + k * 19.6; rbox(ax, x, 48, 16.5, 32, t, fc=c, ec=c, tc="white", fs=8.8, bold=True)
        if k < 4: arrow(ax, (x + 16.9, 64), (x + 19.2, 64), c=INK)
    arrow(ax, (92, 47.5), (92, 28), c=INK, style="-"); arrow(ax, (92, 28), (10, 28), c=INK, style="-"); arrow(ax, (10, 28), (10, 47.5), c=INK)
    ax.text(51, 21, "acrescenta o token ao contexto e repete", ha="center", fontsize=8.8, color=INK)
    rbox(ax, 24, 3, 52, 11, "pára no token de fim ou no limite de tokens", fc="#FEE2E2", ec=RED, fs=8.8, bold=True)
    save(fig, "f06_loop")


def decod():
    p = np.array([0.50, 0.30, 0.10, 0.05, 0.05]); lab = list("ABCDE")
    z = np.log(p)
    def top_k(q, k): q = q.copy(); q[np.argsort(q)[::-1][k:]] = 0; return q / q.sum()
    def top_p(q, pp):
        o = np.argsort(q)[::-1]; c = np.cumsum(q[o]); n = int(np.searchsorted(c, pp - 1e-12) + 1); r = np.zeros_like(q); r[o[:n]] = q[o[:n]]; return r / r.sum()
    paineis = [("original", p), ("T = 0,5", modelos.softmax(z, T=0.5)), ("T = 2", modelos.softmax(z, T=2.0)), ("top-k, k = 2", top_k(p, 2)), ("top-p, p = 0,9", top_p(p, 0.9))]
    fig, axs = plt.subplots(1, 5, figsize=(9.6, 2.7), sharey=True)
    for ax, (t, q) in zip(axs, paineis):
        b = ax.bar(lab, q, color=[IND, CYA, GRN, AMB, PNK], width=0.7); ax.set_title(t, fontsize=9); ax.set_ylim(0, 1)
        for r, v in zip(b, q): ax.text(r.get_x() + r.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
    axs[0].set_ylabel("probabilidade")
    save(fig, "f06_decod")


def qualidade():
    frases = corpus.frases_lm(3000, seed=0); lm = modelos.NgramaLM(3, 0.05).ajustar(frases[:2400]); V = lm.vocab
    def gerar(rng, T=1.0, top_k=None, top_p=None, guloso=False, maxlen=14):
        ctx = []
        for _ in range(maxlen):
            lp = np.log(lm.distribuicao(ctx) + 1e-12); i = int(np.argmax(lp)) if guloso else modelos.amostrar(lp, rng, T=T, top_k=top_k, top_p=top_p); t = V[i]
            if t == "<s>": continue
            ctx.append(t)
            if t == ".": break
        return ctx
    rng = np.random.default_rng(0); cfg = {"gulosa": dict(guloso=True), "T=0,5": dict(T=0.5), "T=1,0": dict(T=1.0), "T=1,5": dict(T=1.5), "T=2,0": dict(T=2.0), "top-k=3": dict(top_k=3), "top-p=0,9": dict(top_p=0.9)}
    pts = {}
    for n, c in cfg.items():
        am = [gerar(rng, **c) for _ in range(300)]; pts[n] = (len({" ".join(s) for s in am}) / 300, np.mean([corpus.frase_valida(s) for s in am]))
    fig, ax = plt.subplots(figsize=(6.2, 4.0)); cols = [GRY, CYA, IND, AMB, RED, GRN, PNK]
    for (n, (dv, va)), c in zip(pts.items(), cols):
        ax.scatter(dv, va, s=90, c=c, zorder=3); off = (-0.08, 0.05) if n == "T=0,5" else (0.015, -0.045); ax.text(dv + off[0], va + off[1], n, fontsize=8.5, color=INK)
    ax.set_xlabel("diversidade (fração de frases distintas)"); ax.set_ylabel("validade (fração de frases válidas)"); ax.set_xlim(-0.05, 1.12); ax.set_ylim(-0.05, 1.1); ax.set_title("Compromisso entre qualidade e diversidade")
    save(fig, "f06_qualidade")


def cache():
    fig, axs = plt.subplots(1, 2, figsize=(9.4, 3.6))
    for ax, titulo, calc in [(axs[0], "Sem cache: recalcula K e V do prefixo", True), (axs[1], "Com cache: calcula só o token novo", False)]:
        ax.set_xlim(0, 6); ax.set_ylim(0, 5.4); ax.axis("off"); ax.set_title(titulo, fontsize=9.5, fontweight="bold")
        for passo in range(4):
            y = 4.2 - passo * 1.15
            for j in range(passo + 1):
                novo = j == passo
                col = AMB if novo else (RED if calc else "#D1FAE5")
                ax.add_patch(FancyBboxPatch((0.9 + j * 0.95, y), 0.85, 0.62, boxstyle="round,pad=0,rounding_size=0.08", fc=col, ec=GRY if not calc and not novo else col, alpha=0.9, lw=1))
                ax.text(0.9 + j * 0.95 + 0.42, y + 0.31, "K,V", ha="center", va="center", fontsize=7, color="white" if (calc and not novo) or novo else INK)
            ax.text(0.1, y + 0.31, f"passo {passo + 1}", fontsize=8, va="center")
        ax.text(3, 0.1, "vermelho = recalculado · âmbar = novo · verde = reaproveitado do cache" if not calc else "cada passo repete o trabalho dos anteriores", ha="center", fontsize=7.6, color=GRY)
    save(fig, "f06_cache")


def contexto():
    fig, ax = canvas(9.2, 2.8, (0, 100), (0, 36)); partes = [("instruções\ndo sistema", 10, IND), ("documentos\nrecuperados", 38, CYA), ("histórico da\nconversa", 24, GRN), ("espaço reservado\npara a resposta", 18, AMB), ("livre", 10, "#E5E7EB")]; x = 2
    for t, w, c in partes:
        rbox(ax, x, 12, w * 0.96, 16, t, fc=c, ec="white", tc="white" if c != "#E5E7EB" else INK, fs=6.6, bold=True, r=0.6); x += w * 0.96
    ax.annotate("", xy=(2, 8), xytext=(x, 8), arrowprops=dict(arrowstyle="<->", color=INK)); ax.text(51, 3.2, "janela de contexto = entrada + saída (em tokens)", ha="center", fontsize=9, color=INK)
    save(fig, "f06_contexto")


if __name__ == "__main__":
    mapa(); loop(); decod(); qualidade(); cache(); contexto()
