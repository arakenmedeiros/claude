"""Figuras do Capítulo 12. As marcadas [D] usam dados calculados com o MLPLM (modelo minúsculo, NumPy); as demais são esquemas."""
import sys; sys.path.insert(0, "/home/user/claude/llm/build"); sys.path.insert(0, "/home/user/claude/llm/codigo/comum")
from fc import *
from llmdidatico import corpus, modelos, avaliacao as av
from llmdidatico.lora import *

_C = {}
def _dados():
    if not _C:
        ant = corpus.frases_lm(3000, seed=0); ant_te = corpus.frases_lm(600, seed=99)
        nov = frases_dominio_novo(1500, seed=1); nov_va = frases_dominio_novo(300, seed=2); nov_te = frases_dominio_novo(600, seed=3)
        base = modelos.MLPLM(vocabulario(ant, nov), ctx=3, d=16, h=64, seed=0); base.treinar(ant, epocas=12, lote=64, lr=0.01)
        _C.update(ant=ant, ant_te=ant_te, nov=nov, nov_va=nov_va, nov_te=nov_te, base=base)
    return _C


def mapa():
    fig, ax = canvas(9.4, 4.4)
    rbox(ax, 2, 66, 26, 26, "Quando ajustar?\nprompt · RAG · ajuste", fc="#E0E7FF", ec=IND, fs=8.8, bold=True)
    rbox(ax, 37, 66, 26, 26, "Dados de ajuste\nformato · divisão\nvazamento · privacidade", fc="#CFFAFE", ec=CYA, fs=8.6, bold=True)
    rbox(ax, 72, 66, 26, 26, "Ajuste completo\n16 bytes por parâmetro", fc="#FEE2E2", ec=RED, fs=8.8, bold=True)
    rbox(ax, 2, 28, 26, 26, "LoRA\nW' = W + (α/r)·B·A", fc="#D1FAE5", ec=GRN, fs=8.8, bold=True)
    rbox(ax, 37, 28, 26, 26, "Quantização\nint8 · int4 · blocos\n→ QLoRA", fc="#FEF3C7", ec=AMB, fs=8.8, bold=True)
    rbox(ax, 72, 28, 26, 26, "Mesclagem e\navaliação antes/depois\nIC · esquecimento", fc="#FCE7F3", ec=PNK, fs=8.6, bold=True)
    rbox(ax, 8, 2, 84, 15, "Riscos e licenças · hiperparâmetros · lista de verificação", fc=LIGHT, ec=GRY, fs=8.8, bold=True)
    arrow(ax, (28.5, 79), (36.5, 79)); arrow(ax, (63.5, 79), (71.5, 79)); arrow(ax, (85, 65.5), (85, 54.5), c=GRY, ls="--")
    arrow(ax, (50, 65.5), (15, 54.5), c=GRY, ls="--"); arrow(ax, (28.5, 41), (36.5, 41)); arrow(ax, (63.5, 41), (71.5, 41)); arrow(ax, (50, 27.5), (50, 17.5), c=GRY, ls="--")
    save(fig, "f12_mapa")


def decisao():
    fig, ax = canvas(9.2, 4.6)
    rbox(ax, 36, 86, 28, 11, "O sistema falha?", fc=IND, ec=IND, tc="white", fs=9, bold=True)
    q = [(2, "Falta informação\n(atual, privada)?", "RAG\n(Cap. 9)", CYA), (35, "Falta instrução ou\nformato claro?", "Prompt\n(Cap. 8)", GRN), (68, "Estilo, formato ou tarefa\nestáveis, com muitos\nexemplos de qualidade?", "Ajuste fino\n(este capítulo)", AMB)]
    for x, t, a, c in q:
        rbox(ax, x, 50, 30, 22, t, fc="white", ec=c, fs=8.4, bold=True); rbox(ax, x, 18, 30, 20, a, fc=c, ec=c, tc="white", fs=9, bold=True)
        arrow(ax, (x + 15, 49.5), (x + 15, 38.5), c=INK); arrow(ax, (50, 85.5), (x + 15, 72.5), c=GRY)
    ax.text(50, 6, "mede-se antes de decidir: conjunto de teste, baseline com prompt e custo por requisição", ha="center", fontsize=8.6, color=GRY, style="italic")
    save(fig, "f12_decisao")


def esquema_lora():
    fig, ax = canvas(9.2, 4.2)
    rbox(ax, 4, 38, 14, 22, "x", fc=LIGHT, ec=GRY, fs=11, bold=True)
    rbox(ax, 30, 62, 28, 24, "W (congelado)\nd × k", fc="#DBEAFE", ec=IND, fs=9.5, bold=True)
    rbox(ax, 28, 14, 14, 24, "A\nr × k", fc="#D1FAE5", ec=GRN, fs=9.5, bold=True); rbox(ax, 46, 14, 14, 24, "B\nd × r", fc="#D1FAE5", ec=GRN, fs=9.5, bold=True)
    ax.text(44, 7, "treináveis · B começa em 0 · escala α/r", ha="center", fontsize=8, color=GRN)
    rbox(ax, 74, 38, 22, 22, "h = Wx + (α/r)·BAx", fc="#FEF3C7", ec=AMB, fs=9.2, bold=True)
    arrow(ax, (18.5, 52), (29.5, 72)); arrow(ax, (18.5, 46), (27.5, 28)); arrow(ax, (42.5, 26), (45.5, 26)); arrow(ax, (58.5, 72), (73.5, 55)); arrow(ax, (60.5, 26), (73.5, 42))
    save(fig, "f12_lora")


def curvas():
    c = _dados(); base = c["base"]
    L = MLPLMLoRA(base, r=4, alfa=8, seed=0); hl = L.treinar(c["nov"], epocas=10, lr=0.005, val=c["nov_va"])
    m, hc = ajuste_completo(base, c["nov"], epocas=10, lr=0.005, val=c["nov_va"])
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.4)); ep = np.arange(1, 11)
    ax[0].plot(ep, hl["val"], "-o", c=GRN, ms=4, label="LoRA r=4 (884 treináveis)"); ax[0].plot(ep, hc["val"], "-s", c=RED, ms=4, label="completo (6781 treináveis)")
    ax[0].axhline(perda_media(base, c["nov_va"]), c=GRY, ls="--", lw=1); ax[0].text(5.5, perda_media(base, c["nov_va"]) - 0.03, "base sem ajuste", color=GRY, fontsize=8)
    ax[0].set_xlabel("época"); ax[0].set_ylabel("perda de validação (tarefa nova)"); ax[0].set_title("Tarefa nova", fontsize=9.5); ax[0].legend(fontsize=7.8, frameon=False)
    # esquecimento ao longo do treino
    ant_te = c["ant_te"]; Lp = MLPLMLoRA(base, r=4, alfa=8, seed=0); pl, pc = [], []
    mm = copy.deepcopy(base); mm.m = {k: np.zeros_like(v) for k, v in mm.P.items()}; mm.v = {k: np.zeros_like(v) for k, v in mm.P.items()}; mm.t = 0
    Xn, yn = Lp.dados(c["nov"]); Xm, ym = mm.dados(c["nov"]); rng1 = np.random.default_rng(0); rng2 = np.random.default_rng(0)
    for e in range(10):
        o1 = rng1.permutation(len(Xn))
        for i in range(0, len(Xn), 64):
            b = o1[i:i + 64]; Lp.passo(Lp.perda_grad(Xn[b], yn[b])[1], 0.005)
        o2 = rng2.permutation(len(Xm))
        for i in range(0, len(Xm), 64):
            b = o2[i:i + 64]; mm.passo(mm.perda_grad(Xm[b], ym[b])[1], 0.005)
        pl.append(Lp.perplexidade(ant_te)); pc.append(mm.perplexidade(ant_te))
    ax[1].plot(ep, pl, "-o", c=GRN, ms=4, label="LoRA r=4"); ax[1].plot(ep, pc, "-s", c=RED, ms=4, label="completo"); ax[1].axhline(base.perplexidade(ant_te), c=GRY, ls="--", lw=1)
    ax[1].text(4, base.perplexidade(ant_te) + 0.15, "base antes do ajuste", color=GRY, fontsize=8)
    ax[1].set_xlabel("época"); ax[1].set_ylabel("perplexidade na tarefa ORIGINAL"); ax[1].set_title("Esquecimento", fontsize=9.5); ax[1].legend(fontsize=7.8, frameon=False)
    fig.suptitle("Ajuste do MLPLM minúsculo (dados sintéticos, não é um LLM)", fontsize=9, color=GRY); save(fig, "f12_curvas")


def efeito_r():
    c = _dados(); base = c["base"]; rs = [1, 2, 4, 8, 16]; nova, orig, npar = [], [], []
    for r in rs:
        res = []
        for s in range(3):
            L = MLPLMLoRA(base, r=r, alfa=2 * r, seed=s); L.treinar(c["nov"], epocas=3, lr=0.005); res.append((L.perplexidade(c["nov_te"]), L.perplexidade(c["ant_te"])))
        mu = np.mean(res, 0); nova.append(mu[0]); orig.append(mu[1]); npar.append(L.n_treinaveis())
    # recuperação de posto numa camada linear (alvo de posto 3)
    rng = np.random.default_rng(0); d_in, d_out = 32, 24; W = rng.normal(0, .3, (d_out, d_in)); U = rng.normal(size=(d_out, 3)); V = rng.normal(size=(3, d_in)); D = 0.5 * U @ V / np.sqrt(3)
    X = rng.normal(size=(512, d_in)); Y = X @ (W + D).T; errs = []
    for r in range(1, 9):
        g = np.random.default_rng(0); A = g.normal(0, 1 / np.sqrt(d_in), (r, d_in)); B = np.zeros((d_out, r)); s = 8 / r
        st = [(A, np.zeros_like(A), np.zeros_like(A)), (B, np.zeros_like(B), np.zeros_like(B))]
        for t in range(1, 3001):
            Z = X @ A.T; E = X @ W.T + s * Z @ B.T - Y; gB = s * 2 / len(X) * E.T @ Z; gA = (s * 2 / len(X) * E @ B).T @ X
            for (P, m, v), G in zip(st, (gA, gB)):
                m *= .9; m += .1 * G; v *= .999; v += .001 * G ** 2; P -= 0.02 * (m / (1 - .9 ** t)) / (np.sqrt(v / (1 - .999 ** t)) + 1e-8)
        errs.append(np.linalg.norm(s * B @ A - D) / np.linalg.norm(D))
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.4))
    ax[0].plot(range(1, 9), errs, "-o", c=IND, ms=4); ax[0].axvline(3, c=AMB, ls="--", lw=1.2); ax[0].text(3.15, 0.5, "posto verdadeiro = 3", color=AMB, fontsize=8)
    ax[0].set_yscale("log"); ax[0].set_xlabel("posto r do adaptador"); ax[0].set_ylabel("erro relativo em ΔW"); ax[0].set_title("Recuperar uma mudança de posto 3", fontsize=9.5)
    ax[1].plot(range(len(rs)), nova, "-o", c=GRN, ms=4, label="perplexidade, tarefa nova"); ax[1].plot(range(len(rs)), orig, "-s", c=RED, ms=4, label="perplexidade, tarefa original")
    ax[1].set_xticks(range(len(rs))); ax[1].set_xticklabels([f"r={r}\n{n}" for r, n in zip(rs, npar)], fontsize=7.5); ax[1].set_xlabel("posto (e nº de parâmetros treináveis)"); ax[1].set_ylabel("perplexidade (média de 3 sementes)")
    ax[1].set_title("MLPLM: efeito de r (3 épocas)", fontsize=9.5); ax[1].legend(fontsize=7.8, frameon=False); save(fig, "f12_posto")


def memoria():
    N = 7e9; n_all = parametros_lora_transformer(4096, 32, 16, d_ff=11008, alvos=("q", "k", "v", "o", "mlp"))
    cen = [("ajuste\ncompleto", estimativa_memoria_gb(N, modo="completo")), ("LoRA\n(base 16 bits)", estimativa_memoria_gb(N, n_all, "lora")), ("QLoRA\n(base 4 bits)", estimativa_memoria_gb(N, n_all, "qlora", dupla=True))]
    fig, ax = plt.subplots(figsize=(6.6, 3.8)); cor = {0: RED, 1: IND, 2: GRN}
    for i, (nome, comp) in enumerate(cen):
        b = 0
        for j, (k, v) in enumerate(comp.items()):
            c = RED if i == 0 else (IND if j == 0 else GRN)
            ax.bar(i, v, bottom=b, color=c, width=0.55, edgecolor="white", label=("pesos, gradientes e otimizador" if i == 0 else ("pesos da base (congelados)" if j == 0 else "adaptadores com otimizador (0,6 GB)")) if i < 2 or j else None)
            b += v
        ax.text(i, b + 2, f"{b:.1f} GB", ha="center", fontsize=9, fontweight="bold")
    h, l = ax.get_legend_handles_labels(); u = dict(zip(l, h)); ax.legend(u.values(), u.keys(), fontsize=7.8, frameon=False, loc="upper right")
    ax.set_xticks(range(3)); ax.set_xticklabels([n for n, _ in cen], fontsize=8.5); ax.set_ylabel("GB (sem ativações)"); ax.set_ylim(0, 128)
    ax.set_title("Modelo hipotético de 7 bilhões de parâmetros", fontsize=9.2); save(fig, "f12_memoria")


def quant():
    c = _dados(); base = c["base"]; W2 = base.P["W2"]; nf = livro_normal(4)
    fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.3))
    blocos = [16, 64, 256, 1024, W2.size]
    for bits, cor in ((8, GRN), (4, IND)):
        ax[0].plot(range(len(blocos)), [erro_reconstrucao(W2, desquantizar_blocos(quantizar_blocos(W2, bits, b)))["sqnr_db"] for b in blocos], "-o", c=cor, ms=4, label=f"int{bits}")
    ax[0].plot(range(len(blocos)), [erro_reconstrucao(W2, desquantizar_livro(quantizar_livro(W2, nf, b)))["sqnr_db"] for b in blocos], "-^", c=AMB, ms=4, label="NF4 didático")
    ax[0].set_xticks(range(len(blocos))); ax[0].set_xticklabels(["16", "64", "256", "1024", "único"], fontsize=8); ax[0].set_xlabel("tamanho do bloco"); ax[0].set_ylabel("SQNR (dB, maior é melhor)"); ax[0].set_title("Erro de reconstrução (W2)", fontsize=9); ax[0].legend(fontsize=7.5, frameon=False)
    ax[1].hist(W2.ravel(), bins=50, color=MID, edgecolor="white"); ax[1].set_title("Distribuição dos pesos", fontsize=9); ax[1].set_xlabel("valor do peso"); ax[1].set_yticks([])
    am = np.abs(W2[:, :1]).max()
    ax[2].eventplot([np.linspace(-1, 1, 15)], lineoffsets=2, colors=IND, linelengths=.6); ax[2].eventplot([nf], lineoffsets=1, colors=AMB, linelengths=.6)
    ax[2].set_yticks([1, 2]); ax[2].set_yticklabels(["NF4\ndidático", "int4\nuniforme"], fontsize=8); ax[2].set_xlabel("valor normalizado (bloco / absmax)"); ax[2].set_title("Níveis de 4 bits", fontsize=9); ax[2].set_ylim(.4, 2.6)
    save(fig, "f12_quant")


def esquecimento():
    c = _dados(); base = c["base"]; L = MLPLMLoRA(base, r=4, alfa=8, seed=0); L.treinar(c["nov"], epocas=10, lr=0.005)
    comp, _ = ajuste_completo(base, c["nov"], epocas=10, lr=0.005)
    mix = MLPLMLoRA(base, r=4, alfa=8, seed=0); mix.treinar(c["nov"] + c["ant"][:1500], epocas=10, lr=0.005)
    ms = [("base", base), ("LoRA r=4", L), ("completo", comp), ("LoRA +\n50% antigos", mix)]
    fig, ax = plt.subplots(figsize=(7.2, 3.6)); x = np.arange(len(ms)); w = 0.36
    o = [m.perplexidade(c["ant_te"]) for _, m in ms]; n = [m.perplexidade(c["nov_te"]) for _, m in ms]
    ax.bar(x - w / 2, o, w, color=RED, label="tarefa original"); ax.bar(x + w / 2, n, w, color=GRN, label="tarefa nova")
    for i in range(len(ms)): ax.text(i - w / 2, o[i] + .08, f"{o[i]:.2f}", ha="center", fontsize=8); ax.text(i + w / 2, n[i] + .08, f"{n[i]:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([a for a, _ in ms], fontsize=8.5); ax.set_ylabel("perplexidade (menor é melhor)"); ax.legend(fontsize=8, frameon=False)
    ax.set_title("Antes/depois: ganho na tarefa nova e custo na original", fontsize=9.5); save(fig, "f12_esquecimento")


def pipeline():
    fig, ax = canvas(9.4, 3.2)
    et = [("Dados\nbrutos", LIGHT, GRY), ("Limpeza,\nPII, dedup.", "#CFFAFE", CYA), ("Divisão\ntreino/val\n/teste", "#E0E7FF", IND), ("Treino\n(LoRA/QLoRA)", "#D1FAE5", GRN), ("Avaliação\nantes/depois\n+ IC", "#FEF3C7", AMB), ("Mesclagem e\nimplantação", "#FCE7F3", PNK)]
    for i, (t, f, e) in enumerate(et):
        rbox(ax, 0.5 + i * 16.6, 45, 15, 34, t, fc=f, ec=e, fs=7, bold=True)
        if i: arrow(ax, (i * 16.6 - 1.1, 62), (i * 16.6 + .6, 62), ms=8)
    ax.text(50, 8, "teste fica trancado até a decisão final · registrar versões de dados, modelo e hiperparâmetros", ha="center", fontsize=8.4, color=GRY, style="italic")
    arrow(ax, (92, 44), (66, 44), c=GRY, ls="--", rad=-0.35); ax.text(79, 22, "regressão? volta ao treino", fontsize=8, color=GRY, ha="center")
    save(fig, "f12_ciclo")


if __name__ == "__main__":
    mapa(); decisao(); esquema_lora(); pipeline(); curvas(); efeito_r(); memoria(); quant(); esquecimento()
