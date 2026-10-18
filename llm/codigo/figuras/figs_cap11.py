"""Figuras do Capítulo 11 (RAG). f11_chunking, f11_recuperacao, f11_limiar, f11_erros e f11_custo são calculadas com
llmdidatico.rag (mesmos procedimentos do texto); f11_mapa, f11_pipeline e f11_injecao são esquemas."""
import sys; sys.path.insert(0, "/home/user/claude/llm/build"); sys.path.insert(0, "/home/user/claude/llm/codigo/comum")
from fc import *
from collections import Counter, defaultdict
from llmdidatico import DOCS, CONSULTAS, avaliacao as av
from llmdidatico.rag import *

RESP = [q for q in CONSULTAS if q[1]]
IX = Indice(fragmentar(DOCS, "documento")); IDF = idf_do_corpus(DOCS)


def mapa():
    fig, ax = canvas(9.4, 4.4)
    rbox(ax, 36, 70, 28, 22, "Pergunta do usuário\n+ identidade e permissões", fc=IND, ec=IND, tc="white", fs=9, bold=True)
    rbox(ax, 3, 36, 26, 22, "Documentos\nnormas · manuais\nmetadados", fc="#E0E7FF", ec=IND, fs=8.6, bold=True)
    rbox(ax, 37, 36, 26, 22, "Recuperação\nBM25 · LSA · híbrido\nfiltros", fc="#CFFAFE", ec=CYA, fs=8.6, bold=True)
    rbox(ax, 71, 36, 26, 22, "Prompt + geração\ndelimitadores · citações\nabstenção", fc="#D1FAE5", ec=GRN, fs=8.6, bold=True)
    arrow(ax, (29.5, 47), (36.5, 47)); arrow(ax, (63.5, 47), (70.5, 47)); arrow(ax, (50, 69.5), (50, 58.5))
    rbox(ax, 3, 4, 28, 20, "Avaliação\nrecall · fidelidade\nabstenção · IC", fc="#FEF3C7", ec=AMB, fs=8.4, bold=True)
    rbox(ax, 36, 4, 28, 20, "Governança\nLGPD · vigência\nexclusão · auditoria", fc="#FCE7F3", ec=PNK, fs=8.4, bold=True)
    rbox(ax, 69, 4, 28, 20, "Segurança\ninjeção · acesso\nvalidação", fc="#FEE2E2", ec=RED, fs=8.4, bold=True)
    for x in (17, 50, 83): arrow(ax, (x, 24.5), (x, 35.5), c=GRY, ls="--")
    save(fig, "f11_mapa")


def pipeline():
    fig, ax = canvas(9.4, 4.0, (0, 100), (0, 60))
    ax.text(1, 56, "Offline (uma vez)", fontsize=8.6, fontweight="bold", color=INK)
    for k, t in enumerate(["Fragmentar\n+ metadados", "Indexar\nBM25 · vetores", "Índice\nversionado"]):
        x = 3 + k * 32; rbox(ax, x, 38, 26, 14, t, fc="#E0E7FF", ec=IND, fs=8.6, bold=True)
        if k < 2: arrow(ax, (x + 26.3, 45), (x + 31.7, 45))
    ax.text(1, 30, "A cada pergunta", fontsize=8.6, fontweight="bold", color=INK)
    etapas = [("Recuperar\n+ filtros", CYA, "recall do\ndocumento"), ("Montar\nprompt", AMB, "cabe no\nlimite?"), ("Gerar\ncom citações", GRN, "correta?\nabstém?"), ("Verificar\n+ registrar", RED, "fiel ao\ncontexto?")]
    for k, (t, c, q) in enumerate(etapas):
        x = 2 + k * 24.5; rbox(ax, x, 12, 21, 14, t, fc="white", ec=c, fs=8.6, bold=True, lw=2)
        ax.text(x + 10.5, 6, q, ha="center", va="center", fontsize=7.4, color=GRY)
        if k < 3: arrow(ax, (x + 21.3, 19), (x + 24.2, 19))
    arrow(ax, (84, 37.6), (14, 26.6), c=GRY, ls="--")
    save(fig, "f11_pipeline")


def chunking():
    por_cat = defaultdict(list)
    for d in DOCS: por_cat[d[1]].append(d)
    LONG = [(f"R{i+1}", c, c, " ".join(f"{d[2]}. {d[3]}" for d in ds)) for i, (c, ds) in enumerate(por_cat.items())]
    nomes, hit, tok = [], [], []
    for nome, tam in [("1 frase", 1), ("2 frases", 2), ("4 frases", 4), ("8 frases", 8), ("regulam.\ninteiro", None)]:
        fr = fragmentar(LONG, "documento") if tam is None else fragmentar(LONG, "frases", tamanho=tam, passo=max(1, tam // 2))
        ix = Indice(fr, usar_titulo=False); h, t = [], []
        for q in RESP:
            ctx = " ".join(ix.fr[i].texto for i, _ in ix.buscar(q[0], "hibrido", 3, um_por_doc=False))
            h.append(float(contem_trecho(ctx, q[2]))); t.append(contar_tokens(ctx))
        nomes.append(nome); hit.append(np.mean(h)); tok.append(np.mean(t))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(nomes, hit, color=IND, alpha=0.85, width=0.55); ax.set_ylim(0, 1.15); ax.set_ylabel("consultas com o trecho da resposta no contexto")
    for i, v in enumerate(hit): ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax2 = ax.twinx(); ax2.plot(nomes, tok, "o-", c=AMB, lw=2); ax2.set_ylabel("tokens médios do contexto (k=3)", color=AMB)
    ax2.spines["right"].set_visible(True); ax2.spines["top"].set_visible(False)
    ax.set_title("Tamanho do fragmento: cobertura e custo", fontsize=9.5); save(fig, "f11_chunking")


def recuperacao():
    fig, ax = plt.subplots(figsize=(6.4, 3.6)); w = 0.26
    for j, (m, c) in enumerate([("bm25", IND), ("lsa", GRN), ("hibrido", AMB)]):
        for i, t in enumerate(["direta", "parafrase"]):
            v = [float(bool(set(ids_docs(IX.buscar(q[0], m, 3), IX)) & set(q[1]))) for q in RESP if q[3] == t]
            mm, lo, hi = av.bootstrap_ic(v, n=2000, seed=0)
            ax.bar(i + (j - 1) * w, mm, w * 0.92, color=c, label=m if i == 0 else None, alpha=0.9)
            ax.errorbar(i + (j - 1) * w, mm, yerr=[[mm - lo], [hi - mm]], c=INK, capsize=3, lw=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["direta (n=38)", "paráfrase (n=8)"]); ax.set_ylim(0, 1.25)
    ax.set_ylabel("recall do documento relevante (k=3)"); ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper right")
    ax.set_title("Recuperação por método e tipo de consulta (IC 95%)", fontsize=9.5); save(fig, "f11_recuperacao")


def divisao(consultas):
    por_tipo = defaultdict(list)
    for q in consultas: por_tipo[q[3]].append(q)
    dev, teste = [], []
    for qs in por_tipo.values():
        for j, q in enumerate(qs): (dev if j % 2 == 0 else teste).append(q)
    return dev, teste


def limiar():
    DEV, _ = divisao(CONSULTAS); lims = np.arange(0, 0.81, 0.1); A, B = [], []
    for lim in lims:
        L = avaliar(PipelineRAG(IX, GeradorSimulado(IDF, limiar=lim), metodo="hibrido", k=3), DEV)
        A.append(np.mean([l["rotulo"] == "ok" for l in L if l["tipo"] != "sem_resposta"]))
        B.append(np.mean([l["rotulo"] == "ok_abstencao" for l in L if l["tipo"] == "sem_resposta"]))
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    ax.plot(lims, A, "o-", c=IND, label="respostas corretas (respondíveis)"); ax.plot(lims, B, "s-", c=RED, label="abstenção correta (sem resposta)")
    ax.axvline(0.5, c=GRY, ls="--"); ax.text(0.51, 0.52, "escolhido\n(DEV)", fontsize=8, color=INK)
    ax.set_xlabel("limiar de cobertura"); ax.set_ylabel("taxa"); ax.set_ylim(0, 1.08); ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("O compromisso entre responder e se abster", fontsize=9.5); save(fig, "f11_limiar")


def erros():
    L = avaliar(PipelineRAG(IX, GeradorSimulado(IDF, limiar=0.5), metodo="hibrido", k=3), CONSULTAS)
    rot = [("ok", GRN, "correta"), ("ok_abstencao", "#34D399", "abstenção correta"), ("geracao_errada", AMB, "geração errada"),
           ("abstencao_indevida", VIO, "abstenção indevida"), ("falha_recuperacao", RED, "falha de recuperação"), ("alucinacao_sem_resposta", "#7F1D1D", "resposta sem suporte")]
    tipos = ["direta", "parafrase", "sem_resposta"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4)); base = np.zeros(3)
    for r, c, nome in rot:
        v = np.array([sum(l["rotulo"] == r for l in L if l["tipo"] == t) for t in tipos], float)
        if v.sum() == 0: continue
        ax.barh(range(3), v, left=base, color=c, label=nome, height=0.55)
        for i in range(3):
            if v[i] > 0: ax.text(base[i] + v[i] / 2, i, int(v[i]), ha="center", va="center", color="white", fontsize=8, fontweight="bold")
        base += v
    ax.set_yticks(range(3)); ax.set_yticklabels(["direta (38)", "paráfrase (8)", "sem resposta (10)"]); ax.invert_yaxis()
    ax.set_xlabel("número de consultas"); ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax.set_title("Desfechos por tipo de consulta (limiar 0,5)", fontsize=9.5); save(fig, "f11_erros")


def injecao():
    fig, ax = canvas(9.0, 3.9, (0, 100), (0, 60))
    rbox(ax, 2, 40, 22, 14, "Documento\nnão confiável", fc="#FEE2E2", ec=RED, fs=8.6, bold=True)
    cam = [("1. Controle de\nentrada no índice", "origem · autor · revisão"), ("2. Sanitizar e\ndelimitar", "<<<DOC>>> · escape"), ("3. Validar a\nsaída", "citação · trecho · formato"), ("4. Limitar o\nque o sistema faz", "menor privilégio")]
    for k, (t, d) in enumerate(cam):
        x = 27 + k * 18.5; rbox(ax, x, 40, 16, 14, t, fc="#E0E7FF", ec=IND, fs=7.0, bold=True)
        ax.text(x + 8, 34.5, d, ha="center", fontsize=6.6, color=GRY)
    arrow(ax, (24.3, 47), (26.7, 47), lw=1.2, ms=9)
    for k in range(3): arrow(ax, (27 + k * 18.5 + 16.2, 47), (27 + (k + 1) * 18.5 - 0.2, 47), lw=1.2, ms=9)
    rbox(ax, 8, 6, 84, 14, "Envenenamento de conteúdo: um fato falso, sem instrução alguma, atravessa as camadas 2 e 3;\ncontrole de origem, vigência, revisão e monitoramento são a defesa.", fc="#FEF3C7", ec=AMB, fs=8)
    arrow(ax, (50, 21), (50, 32), c=AMB, ls="--")
    save(fig, "f11_injecao")


def custo():
    ks = [1, 3, 5, 10]; rec, tk = [], []
    for k in ks:
        p = PipelineRAG(IX, GeradorSimulado(IDF, limiar=0.5), metodo="hibrido", k=k)
        rs = [p.responder(q[0]) for q in RESP]
        rec.append(np.mean([float(bool(set(r["incluidos"]) & set(q[1]))) for r, q in zip(rs, RESP)])); tk.append(np.mean([contar_tokens(r["prompt"]) for r in rs]))
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    ax.plot(ks, rec, "o-", c=IND, lw=2); ax.set_ylim(0.8, 1.0); ax.set_xlabel("k (fragmentos no contexto)"); ax.set_ylabel("recall do documento no contexto", color=IND)
    ax2 = ax.twinx(); ax2.plot(ks, tk, "s--", c=AMB, lw=2); ax2.set_ylabel("tokens de entrada (média)", color=AMB)
    ax2.spines["right"].set_visible(True); ax2.spines["top"].set_visible(False); ax.set_xticks(ks)
    for k, r, t in zip(ks, rec, tk): ax.text(k, r + 0.006, f"{r:.2f}", ha="center", fontsize=8, color=IND)
    ax.set_title("Mais contexto: mais recall, mais custo", fontsize=9.5); save(fig, "f11_custo")


if __name__ == "__main__":
    mapa(); pipeline(); chunking(); recuperacao(); limiar(); erros(); injecao(); custo()
