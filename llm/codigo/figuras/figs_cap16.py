"""Figuras do Capítulo 16. As figuras [D] usam dados de SIMULADORES (SimuladorLLM, gerador extrativo, juiz simulado), não de LLMs reais."""
import sys; sys.path.insert(0, "/home/user/claude/llm/build"); sys.path.insert(0, "/home/user/claude/llm/codigo/comum")
import re, json, itertools
from fc import *
from sklearn.metrics import cohen_kappa_score
from llmdidatico import corpus, DOCS, CONSULTAS, avaliacao as av
from llmdidatico.modelos import BM25, tokenizar_sem_stop
from llmdidatico.simulador import SimuladorLLM
from llmdidatico.aval_llm import ic_wilson, JuizEnviesado, tokens


# ------------------------------------------------------------------ conceituais
def mapa():
    fig, ax = canvas(9.4, 4.4)
    rbox(ax, 2, 66, 28, 26, "Desenho do teste\nestratos · casos difíceis\ntamanho · dev/teste", fc="#E0E7FF", ec=IND, fs=8.6, bold=True)
    rbox(ax, 36, 66, 28, 26, "Métricas\nclassificação · extração\ntexto aberto · RAG", fc="#CFFAFE", ec=CYA, fs=8.6, bold=True)
    rbox(ax, 70, 66, 28, 26, "Juízes\nLLM como juiz · humanos\nkappa · alfa · calibração", fc="#D1FAE5", ec=GRN, fs=8.6, bold=True)
    arrow(ax, (30.5, 79), (35.5, 79), c=INK); arrow(ax, (64.5, 79), (69.5, 79), c=INK)
    rbox(ax, 2, 28, 44, 26, "Comparação com incerteza\nIC · McNemar · bootstrap pareado\nmúltiplas comparações · variância", fc="#FEF3C7", ec=AMB, fs=8.6, bold=True)
    rbox(ax, 54, 28, 44, 26, "Depois da entrega\nregressão contínua · monitoramento\namostragem · feedback", fc="#FCE7F3", ec=PNK, fs=8.6, bold=True)
    arrow(ax, (50, 65.5), (30, 55), c=GRY, ls="--"); arrow(ax, (84, 65.5), (74, 55), c=GRY, ls="--"); arrow(ax, (46.5, 41), (53.5, 41), c=INK)
    rbox(ax, 2, 2, 96, 17, "Relato responsável: cartão de avaliação\nversão · semente · data · teste · intervalos · limitações", fc=NAVY, ec=NAVY, tc="white", fs=8.8, bold=True)
    arrow(ax, (26, 27.5), (26, 19.5), c=GRY, ls="--"); arrow(ax, (76, 27.5), (76, 19.5), c=GRY, ls="--")
    save(fig, "f16_mapa")


def tipos():
    fig, ax = canvas(9.0, 4.2, (0, 100), (0, 62))
    degraus = [("Intrínseca: perplexidade", "barata, rápida; acompanha o treinamento;\nnão comparável entre tokenizadores", IND),
               ("Extrínseca: tarefa", "acerto, F1, fidelidade, abstenção\nem casos de teste do seu problema", CYA),
               ("Com usuários", "satisfação, reabertura de chamado,\ntempo economizado, erros graves", GRN),
               ("No negócio ou no serviço", "custo, risco, equidade, conformidade", AMB)]
    for k, (t, d, c) in enumerate(degraus):
        x = 2 + k * 6; y = 46 - k * 14
        rbox(ax, x, y, 30, 11, t, fc=c, ec=c, tc="white", fs=8.6, bold=True); rbox(ax, x + 32, y, 60 - k * 6, 11, d, fc="white", ec=c, fs=8.2, ha="left")
    arrow(ax, (96, 8), (96, 56), c=GRY, lw=1.6); ax.text(97.5, 32, "mais caro,\nmais próximo\ndo que importa", fontsize=7.8, color=GRY, rotation=90, ha="center", va="center")
    save(fig, "f16_tipos")


def protocolo():
    fig, ax = canvas(9.4, 3.9)
    rbox(ax, 1, 60, 17, 30, "Par de\nrespostas\n(A, B)", fc=IND, ec=IND, tc="white", fs=9, bold=True)
    rbox(ax, 26, 66, 24, 24, "Juiz: A depois B\n→ vencedor 1", fc="white", ec=CYA, fs=8.4, bold=True)
    rbox(ax, 26, 28, 24, 24, "Juiz: B depois A\n→ vencedor 2", fc="white", ec=CYA, fs=8.4, bold=True)
    arrow(ax, (18.4, 78), (25.6, 78), c=INK); arrow(ax, (18.4, 68), (25.6, 42), c=INK)
    rbox(ax, 58, 48, 20, 24, "Concordam?\nsim: vencedor\nnão: empate", fc="#FEF3C7", ec=AMB, fs=8.4, bold=True)
    arrow(ax, (50.4, 78), (57.6, 64), c=INK); arrow(ax, (50.4, 40), (57.6, 56), c=INK)
    rbox(ax, 84, 48, 15, 24, "Resultado\ncom\nempates", fc=GRN, ec=GRN, tc="white", fs=8.4, bold=True); arrow(ax, (78.4, 60), (83.6, 60), c=INK)
    rbox(ax, 40, 2, 58, 18, "Amostra revisada por humanos: kappa/alfa e calibração\n(sensibilidade e especificidade do juiz)", fc="#FCE7F3", ec=PNK, fs=8.4, bold=True)
    arrow(ax, (88, 47.5), (88, 20.5), c=GRY, ls="--")
    save(fig, "f16_protocolo")


def ci():
    fig, ax = canvas(9.4, 3.6)
    cx = [("Mudança\nprompt · modelo\nbase · código", IND), ("Testes\nautomáticos\nmesmos casos", CYA), ("Portão\nIC da diferença\n+ estratos", AMB), ("Relatório\narquivado\n(versão, semente)", VIO), ("Implantação\ngradual", GRN)]
    for k, (t, c) in enumerate(cx):
        x = 1 + k * 20; rbox(ax, x, 48, 17, 32, t, fc=c, ec=c, tc="white", fs=8.2, bold=True)
        if k < 4: arrow(ax, (x + 17.4, 64), (x + 19.6, 64), c=INK)
    rbox(ax, 14, 6, 72, 20, "Produção: monitoramento, amostra de revisão humana e feedback\nrealimentam o conjunto de teste", fc="#F3F4F6", ec=GRY, fs=8.6)
    arrow(ax, (90, 47.5), (86, 26.5), c=GRY, ls="--"); arrow(ax, (14, 16), (10, 47.5), c=GRY, ls="--", rad=-0.2)
    ax.text(50, 36, "reprova: volta para corrigir", fontsize=7.8, color=RED, ha="center"); 
    save(fig, "f16_ci")


def benchmark():
    t = np.linspace(0, 10, 200)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    nota = 1 / (1 + np.exp(-(t - 4.2) * 1.2)) * 0.97
    ax.plot(t, nota, c=IND, lw=2.2, label="nota dos melhores modelos")
    ax.axhline(0.97, c=GRY, ls=":", lw=1); ax.text(0.2, 0.985, "teto", fontsize=8, color=GRY)
    ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
    ax2.plot(t, 1 - np.exp(-t / 4), c=RED, lw=1.8, ls="--", label="risco de contaminação")
    ax.axvspan(6.5, 10, color=AMB, alpha=0.15); ax.text(8.2, 0.3, "saturado:\ndiscrimina pouco", ha="center", fontsize=8.4)
    ax.set_xlabel("tempo desde a publicação (ilustrativo)"); ax.set_ylabel("nota (ilustrativa)"); ax2.set_ylabel("risco (ilustrativo)")
    ax.set_xticks([]); ax.set_yticks([]); ax2.set_yticks([])
    l1, n1 = ax.get_legend_handles_labels(); l2, n2 = ax2.get_legend_handles_labels(); ax.legend(l1 + l2, n1 + n2, loc="center left", fontsize=8, frameon=False)
    ax.set_title("Ciclo de vida de um benchmark (curva ilustrativa)", fontsize=9.5)
    save(fig, "f16_benchmark")


# ------------------------------------------------------------------ baseadas em dados
def amostra():
    ns = np.arange(10, 1501, 10); fig, ax = plt.subplots(figsize=(5.8, 3.6))
    for p, c in ((0.5, RED), (0.8, IND), (0.95, GRN)):
        larg = [ic_wilson(round(p * n), n)[1] - ic_wilson(round(p * n), n)[0] for n in ns]
        ax.plot(ns, np.array(larg) * 100, c=c, lw=2, label=f"acerto verdadeiro = {p:.2f}")
    for n, l in ((100, "100"), (400, "400"), (1500, "1500")): ax.axvline(n, c=MID, lw=0.8, zorder=0)
    ax.set_xlabel("número de casos de teste"); ax.set_ylabel("largura do IC 95% (pontos percentuais)"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("Quanto mais casos, mais estreito o intervalo (Wilson)", fontsize=9.5)
    save(fig, "f16_amostra")


def rag():
    ids = [d[0] for d in DOCS]; textos = [f"{d[2]}. {d[3]}" for d in DOCS]; por_id = {d[0]: d[3] for d in DOCS}
    bm = BM25([tokenizar_sem_stop(t) for t in textos])
    def recuperar(q, k=3):
        s = bm.pontuar(tokenizar_sem_stop(q)); o = np.argsort(-s, kind="stable")[:k]; return [ids[i] for i in o], [float(s[i]) for i in o]
    def sentencas(t): return [s.strip() for s in re.split(r"(?<=[.;])\s+", t) if s.strip()]
    def gerar(q, ctx, top, tau):
        if top < tau: return "não consta"
        qt = set(tokenizar_sem_stop(q)); melhor, bs = "", -1
        for r, i in enumerate(ctx):
            for s in sentencas(por_id[i]):
                sc = len(qt & set(tokenizar_sem_stop(s))) - 0.01 * r
                if sc > bs: melhor, bs = s, sc
        return melhor
    contem = lambda tr, rs: " ".join(tokens(tr)) in " ".join(tokens(rs))
    res = {}
    for q, rel, tr, tp in CONSULTAS:
        ctx, sc = recuperar(q); resp = gerar(q, ctx, sc[0], 4.5)
        res.setdefault(tp, []).append((bool(set(ctx) & set(rel)) if rel else None, resp == "não consta" if tp == "sem_resposta" else contem(tr, resp)))
    fig, ax = plt.subplots(figsize=(6.2, 3.6)); w = 0.26
    for j, (tp, nome) in enumerate((("direta", "direta"), ("parafrase", "paráfrase"), ("sem_resposta", "sem resposta"))):
        l = res[tp]
        if tp == "sem_resposta":
            ax.bar(j + w, np.mean([o for _, o in l]), w, color=GRN); ax.text(j + w, np.mean([o for _, o in l]) + 0.02, f"{np.mean([o for _, o in l]):.2f}", ha="center", fontsize=8)
        else:
            a = np.mean([r for r, _ in l]); b = np.mean([o for _, o in l]); c = np.mean([o for r, o in l if r])
            for dx, v, col in ((-w, a, IND), (0, b, AMB), (w, c, CYA)): ax.bar(j + dx, v, w, color=col); ax.text(j + dx, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax.bar(0, 0, color=IND, label="recuperou (top 3)"); ax.bar(0, 0, color=AMB, label="resposta correta"); ax.bar(0, 0, color=CYA, label="correta dado que recuperou"); ax.bar(0, 0, color=GRN, label="recusa correta")
    ax.set_xticks(range(3)); ax.set_xticklabels(["direta\n(n=38)", "paráfrase\n(n=8)", "sem resposta\n(n=10)"]); ax.set_ylim(0, 1.18); ax.set_ylabel("fração")
    ax.legend(frameon=False, fontsize=7.6, ncol=2, loc="upper right"); ax.set_title("RAG simulado: onde está a falha? (não é um LLM)", fontsize=9.5)
    save(fig, "f16_rag")


def juiz():
    rng = np.random.default_rng(7)
    resp = lambda q, p=None, e=None: {"qualidade": q, "palavras": int(p if p is not None else np.clip(60 + 12 * q + rng.normal(0, 20), 15, 300)), "estilo": e or rng.choice(["juiz", "outro"])}
    iguais = [(resp(2, 80, "outro"), resp(2, 80, "outro")) for _ in range(500)]
    vps = (0.0, 0.3, 0.6, 1.0); pri = [np.mean([JuizEnviesado(vies_posicao=vp, semente=1).comparar(a, b) == 1 for a, b in iguais[:1]]) for vp in vps]
    pri = []
    for vp in vps:
        j = JuizEnviesado(vies_posicao=vp, semente=1); pri.append(np.mean([j.comparar(a, b) == 1 for a, b in iguais]))
    des = []
    while len(des) < 600:
        a, b = resp(int(rng.integers(0, 4))), resp(int(rng.integers(0, 4)))
        if a["qualidade"] != b["qualidade"]: des.append((a, b))
    hum = np.array(["a" if a["qualidade"] > b["qualidade"] else "b" for a, b in des]); hum = np.where(rng.random(len(hum)) < 0.06, np.where(hum == "a", "b", "a"), hum)
    conc = []
    for troca in (False, True):
        j = JuizEnviesado(vies_posicao=0.8, semente=3); v = np.array([j.comparar_protocolo(a, b, troca) for a, b in des]); dec = v != "empate"; conc.append((np.mean(v[dec] == hum[dec]), dec.mean()))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.2, 3.4), gridspec_kw={"width_ratios": [1.2, 1]})
    a1.plot(vps, pri, "o-", c=RED, lw=2); a1.axhline(0.5, c=GRY, ls="--", lw=1); a1.text(0.55, 0.455, "sem viés: 50%", fontsize=8, color=GRY)
    for x, y in zip(vps, pri): a1.text(x, y + 0.03, f"{y:.2f}", ha="center", fontsize=8)
    a1.set_ylim(0.3, 1.1); a1.set_xlabel("viés de posição programado"); a1.set_ylabel("fração que prefere a 1ª resposta"); a1.set_title("Pares de qualidade idêntica", fontsize=9)
    a2.bar([0, 1], [c[0] for c in conc], color=[GRY, GRN], width=0.55); a2.set_xticks([0, 1]); a2.set_xticklabels(["sem troca\nde ordem", "com troca\nde ordem"])
    for i, c in enumerate(conc): a2.text(i, c[0] + 0.015, f"{c[0]:.2f}\n({c[1]*100:.0f}% decididos)", ha="center", fontsize=8)
    a2.set_ylim(0, 1.15); a2.set_ylabel("concordância com humanos"); a2.set_title("Mitigação por troca de ordem", fontsize=9)
    fig.suptitle("Juiz SIMULADO por regras (não é um LLM)", fontsize=9.5, y=1.02); save(fig, "f16_juiz")


def concordancia():
    rng = np.random.default_rng(11); N = 150; verdade = rng.integers(0, 4, N)
    def anot(p):
        r = verdade.copy(); err = rng.random(N) > p; r[err] = np.clip(r[err] + rng.choice([-1, 1], err.sum()), 0, 3); return r
    H = [anot(0.80), anot(0.75), anot(0.70)]
    juiz = JuizEnviesado(vies_verbosidade=0.6, ruido=0.45, semente=2)
    pal = np.clip(60 + 12 * verdade + rng.normal(0, 20, N), 15, 300)
    notas = np.array([juiz.nota({"qualidade": int(q), "palavras": int(p), "estilo": "outro"}) for q, p in zip(verdade, pal)])
    J = np.clip(np.round(notas), 0, 3).astype(int)
    A = H + [J]; nomes = ["H1", "H2", "H3", "Juiz\n(sim.)"]
    M = np.eye(4)
    for i, j in itertools.combinations(range(4), 2): M[i, j] = M[j, i] = cohen_kappa_score(A[i], A[j], weights="quadratic")
    fig, ax = plt.subplots(figsize=(4.8, 4.0)); im = ax.imshow(M, cmap="Blues", vmin=0.5, vmax=1)
    ax.set_xticks(range(4)); ax.set_xticklabels(nomes); ax.set_yticks(range(4)); ax.set_yticklabels(nomes)
    for i in range(4):
        for j in range(4): ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", color="white" if M[i, j] > 0.8 else INK, fontsize=9)
    for s in ax.spines.values(): s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.046, label="kappa ponderado quadrático"); ax.set_title("Concordância entre avaliadores simulados", fontsize=9.5)
    save(fig, "f16_concordancia")


def variancia():
    X, y = corpus.intencoes(100, seed=31)
    def acc(semente, ordem):
        sim = SimuladorLLM("v2", semente=semente); ok = np.zeros(len(X))
        for i in ordem:
            m = re.fullmatch(r"rotulo: (\w+)", sim.responder(X[i]).strip()); ok[i] = float(bool(m) and m.group(1) == y[i])
        return ok.mean()
    base = np.arange(len(X)); rng = np.random.default_rng(0)
    por_semente = [acc(s, base) for s in range(20)]; por_ordem = [acc(0, rng.permutation(len(X))) for _ in range(20)]
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from llmdidatico.modelos import amostrar
    Xtr, ytr = corpus.intencoes(40, seed=1); vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(Xtr)
    clf = LogisticRegression(max_iter=300).fit(vec.transform(Xtr), ytr); lg = clf.decision_function(vec.transform(X)); cls = list(clf.classes_); yv = np.array([cls.index(r) for r in y])
    temps = {}
    for T in (0.25, 0.5, 1.0, 2.0):
        temps[T] = [np.mean(np.array([amostrar(l, np.random.default_rng(100 + k), T=T) for l in lg]) == yv) for k in range(20)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.5), gridspec_kw={"width_ratios": [1, 1.3]})
    for i, (d, nm, c) in enumerate(((por_semente, "20 sementes", IND), (por_ordem, "20 ordens", CYA))):
        a1.boxplot([d], positions=[i], widths=0.5, showfliers=False, boxprops=dict(color=c), medianprops=dict(color=INK), whiskerprops=dict(color=c), capprops=dict(color=c))
        a1.scatter(np.full(len(d), i) + np.random.default_rng(i).uniform(-0.12, 0.12, len(d)), d, s=12, color=c, zorder=3)
    a1.set_xticks([0, 1]); a1.set_xticklabels(["20 sementes", "20 ordens\ndos casos"]); a1.set_ylabel("acerto da v2 (simulador)"); a1.set_title("Simulador v2", fontsize=9)
    ks = list(temps)
    a2.boxplot([temps[k] for k in ks], positions=range(len(ks)), widths=0.5, showfliers=False, boxprops=dict(color=VIO), medianprops=dict(color=INK), whiskerprops=dict(color=VIO), capprops=dict(color=VIO))
    for i, k in enumerate(ks): a2.scatter(np.full(20, i) + np.random.default_rng(i).uniform(-0.12, 0.12, 20), temps[k], s=12, color=VIO, zorder=3)
    a2.set_xticks(range(len(ks))); a2.set_xticklabels([f"T={k}" for k in ks]); a2.set_ylabel("acerto"); a2.set_title("Classificador de apoio amostrado (não é LLM)", fontsize=9)
    save(fig, "f16_variancia")


def monitor():
    rng = np.random.default_rng(0); n = 300; taxa = [0.08] * 8 + [0.15] * 4; c = rng.binomial(n, taxa); p = c / n
    pr = c[:6].sum() / (6 * n); dp = np.sqrt(pr * (1 - pr) / n); lim = (max(0, pr - 3 * dp), pr + 3 * dp)
    fig, ax = plt.subplots(figsize=(6.4, 3.4)); s = np.arange(1, 13)
    ax.axhspan(*lim, color=MID, alpha=0.6, label="limites de 3 desvios"); ax.axhline(pr, c=GRY, ls="--", lw=1, label="referência (semanas 1-6)")
    ax.plot(s, p, "o-", c=IND, lw=1.8); al = (p > lim[1]) | (p < lim[0]); ax.scatter(s[al], p[al], c=RED, s=60, zorder=4, label="alerta")
    ax.axvline(8.5, c=AMB, ls=":", lw=1.4); ax.text(8.6, 0.215, "mudança\nsimulada", fontsize=8, color=AMB)
    ax.set_xlabel("semana"); ax.set_ylabel("taxa de abstenção"); ax.set_ylim(0, 0.24); ax.set_xticks(s); ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Carta de controle (dados SIMULADOS)", fontsize=9.5); save(fig, "f16_monitor")


if __name__ == "__main__":
    mapa(); tipos(); protocolo(); ci(); benchmark(); amostra(); rag(); juiz(); concordancia(); variancia(); monitor()
