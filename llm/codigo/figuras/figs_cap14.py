"""Figuras do Capítulo 14. As figuras [D] usam o agente didático (SIMULAÇÃO BASEADA EM REGRAS, não um LLM);
os números descrevem o simulador. Experimentos: 87 tarefas sintéticas x 20 sementes."""
import sys; sys.path.insert(0, "/home/user/claude/llm/build"); sys.path.insert(0, "/home/user/claude/llm/codigo/comum")
from fc import *
from collections import Counter
from llmdidatico.agente import (Falhas, avaliar, gerar_tarefas, kb_com_envenenados, Politica, Ambiente, criar_registro,
                                fluxo_roteamento, politica_para)
from llmdidatico import avaliacao as av

KB = kb_com_envenenados(); TAREFAS = gerar_tarefas(); K = 20


def ruido(p, rec=0.7):
    return Falhas(p_arg_invalido=p / 2, p_ferramenta_errada=p / 4, p_laco=p / 4, p_recupera_erro=rec)


def mapa():
    fig, ax = canvas(9.4, 4.6)
    itens = [(1, 70, "Ferramentas\nesquema · validação\ntempo limite · idempotência", CYA), (34, 76, "Fluxos\nencadeamento · roteamento\nparalelização · orquestrador\navaliador–otimizador", GRN),
             (68, 70, "Memória e plano\ncurta · longa · resumo", VIO), (1, 8, "Segurança\nmenor privilégio · confirmação\ninjeção indireta · sandbox", RED),
             (34, 2, "Avaliação\nsucesso · passos · custo · IC", AMB), (68, 8, "Integração\nAPI de LLM · protocolo\nde ferramentas (MCP)", PNK)]
    for x, y, t, c in itens:
        arrow(ax, (x + 15, y + (0 if y > 50 else 20)), (50, 50 + (-6 if y > 50 else 6)), c=GRY, ls="--", lw=1.1, ms=8)
    rbox(ax, 30, 42, 40, 16, "Agente\nobservar–decidir–agir", fc=IND, ec=IND, tc="white", fs=10, bold=True)
    for x, y, t, c in itens:
        rbox(ax, x, y, 31, 22 if y > 50 else 20, t, fc="white", ec=c, fs=7.4, bold=True, lw=1.8)
    save(fig, "f14_mapa")


def chamada():
    fig, ax = canvas(9.0, 4.8)
    cols = [(17, "Aplicação (código)", IND), (52, "Modelo", AMB), (87, "Ferramenta", GRN)]
    for x, t, c in cols:
        rbox(ax, x - 12, 92, 24, 7, t, fc=c, ec=c, tc="white", fs=9, bold=True)
        ax.plot([x, x], [6, 92], color=GRY, lw=1, ls=":")
    A, M, F = 17, 52, 87
    setas = [(A, M, 82, "1  pergunta + esquemas das ferramentas"), (M, A, 70, "2  PROPÕE: tool_use(nome, argumentos)"), (A, F, 46, "4  EXECUTA a função"),
             (F, A, 36, "5  resultado ou erro"), (A, M, 24, "6  tool_result (dado, não instrução)"), (M, A, 12, "7  resposta final ou nova proposta")]
    for x1, x2, y, t in setas:
        arrow(ax, (x1 + (1 if x2 > x1 else -1), y), (x2 + (-1 if x2 > x1 else 1), y), c=INK, lw=1.3, ms=10)
        ax.text((x1 + x2) / 2, y + 2.8, t, ha="center", fontsize=7.4)
    rbox(ax, 2, 54, 30, 8, "3  valida o esquema, autoriza,\npede confirmação se preciso", fc="#E0E7FF", ec=IND, fs=7)
    save(fig, "f14_chamada")


def ciclo():
    fig, ax = canvas(8.8, 4.4)
    for x, y, t, c in [(2, 62, "Observar\nresultado anterior,\nmemória, tarefa", CYA), (35, 62, "Decidir (modelo):\nchamar ferramenta\nou responder", AMB), (68, 62, "Agir\ncódigo valida,\nautoriza e executa", GRN)]:
        rbox(ax, x, y, 30, 28, t, fc="white", ec=c, fs=8.2, bold=True, lw=2)
    arrow(ax, (32.3, 76), (34.7, 76), c=INK); arrow(ax, (65.3, 76), (67.7, 76), c=INK)
    arrow(ax, (83, 61.5), (17, 61.5), c=INK, rad=-0.18)
    ax.text(50, 42, "repete até uma condição de parada", ha="center", fontsize=8.2, style="italic")
    rbox(ax, 2, 0, 46, 38, "Paradas\n• resposta final\n• limite de passos\n• orçamento de tokens\n• laço detectado\n• encaminhar a um humano", fc="#FEF3C7", ec=AMB, fs=7.8, ha="left")
    rbox(ax, 52, 0, 46, 38, "Guardrails em cada volta\n• menor privilégio\n• confirmação humana\n• sanitizar saída da ferramenta\n• registro de auditoria", fc="#FCE7F3", ec=PNK, fs=7.8, ha="left")
    save(fig, "f14_ciclo")


def fluxos():
    fig, ax = canvas(9.2, 4.4)
    def painel(x, y, titulo, c):
        rbox(ax, x, y, 29, 40, "", fc="white", ec=c, lw=1.6); ax.text(x + 14.5, y + 36, titulo, ha="center", fontsize=8.6, fontweight="bold", color=c)
    def cx(x, y, t, w=7, h=7):
        rbox(ax, x, y, w, h, t, fc=LIGHT, ec=GRY, fs=6.8)
    P = [(2, 55, "Encadeamento", IND), (35, 55, "Roteamento", GRN), (68, 55, "Paralelização", AMB), (2, 6, "Orquestrador–trab.", VIO), (35, 6, "Avaliador–otimizador", RED)]
    for x, y, t, c in P: painel(x, y, t, c)
    for i in range(3): cx(4.5 + i * 9, 68, f"etapa {i+1}", 7.4, 8)
    for i in range(2): arrow(ax, (11.9 + i * 9, 72), (13.5 + i * 9, 72), c=INK, ms=6)
    cx(38, 66, "classifica", 9, 8); [cx(55, 59 + 9 * j, f"rota {j+1}", 7.5, 6) for j in range(3)]; [arrow(ax, (47.2, 70), (54.8, 62 + 9 * j), c=GRY, ms=6, lw=1) for j in range(3)]
    for j in range(3): cx(71, 59 + 9 * j, f"tarefa {j+1}", 9, 6)
    cx(84, 68, "junta", 6.5, 6)
    cx(10, 28, "orquestrador", 14, 7); [cx(5 + 9 * j, 12, f"T{j+1}", 6, 6) for j in range(3)]; [arrow(ax, (17 + 0, 28), (8 + 9 * j, 18.2), c=GRY, ms=6, lw=1) for j in range(3)]
    cx(38, 28, "gerador", 10, 8); cx(53, 28, "avaliador", 10, 8); arrow(ax, (48.2, 32), (52.8, 32), c=INK, ms=7); arrow(ax, (58, 27.6), (43, 27.6), c=RED, rad=-0.5, ms=7); ax.text(49, 10, "feedback até aprovar\nou atingir o limite", ha="center", fontsize=7)
    ax.text(83, 26, "Em todos: o CÓDIGO\ndefine o caminho.\nNo agente, é o modelo\nquem escolhe.", ha="center", fontsize=8, style="italic")
    save(fig, "f14_fluxos")


def camadas():
    fig, ax = canvas(9.4, 3.4, (0, 100), (0, 50))
    cam = [("Saída da\nferramenta\n(não confiável)", RED), ("Sanitizar e\nmarcar como\ndado", AMB), ("Modelo\npropõe\numa ação", MID), ("Ferramenta\npermitida?\n(privilégio)", GRN),
           ("Argumentos\ne destino\nválidos?", CYA), ("Confirmação\nhumana", VIO), ("Executar\ne auditar", IND)]
    w = 12.6
    for i, (t, c) in enumerate(cam):
        x = 1 + i * 14
        rbox(ax, x, 12, w, 28, t, fc=c, ec=c, tc="white" if c not in (MID, AMB) else INK, fs=6.8, bold=True)
        if i < len(cam) - 1: arrow(ax, (x + w + 0.1, 26), (x + 14 - 0.1, 26), c=INK, ms=7, lw=1.1)
    ax.text(50, 4, "Camadas independentes: se uma falhar, outra pode conter o dano (nenhuma elimina o risco)", ha="center", fontsize=8.2, style="italic")
    save(fig, "f14_camadas")


def f_ruido():
    ps = [0, 0.2, 0.4, 0.6, 0.8]; M, L, H, S, T = [], [], [], [], []
    for p in ps:
        r = avaliar(TAREFAS, ruido(p), KB, K=K, max_passos=6)
        m, lo, hi = av.bootstrap_ic(r["taxa"], n=1000, seed=0); M.append(m); L.append(lo); H.append(hi); S.append(r["passos"].mean()); T.append(r["tokens"].mean())
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.6, 3.4))
    a.fill_between(ps, L, H, color=CYA, alpha=0.3); a.plot(ps, M, "o-", c=IND); a.set_ylim(0.5, 1); a.set_xlabel("intensidade p das falhas injetadas"); a.set_ylabel("taxa de sucesso (média das tarefas)"); a.set_title("Sucesso e IC 95% (bootstrap)", fontsize=9.5)
    b.plot(ps, S, "o-", c=AMB, label="passos"); b.set_xlabel("intensidade p das falhas injetadas"); b.set_ylabel("passos por tarefa", color=AMB)
    b2 = b.twinx(); b2.plot(ps, T, "s--", c=GRN); b2.set_ylabel("tokens de entrada (proxy)", color=GRN); b2.spines["right"].set_visible(True); b.set_title("Custo cresce com as falhas", fontsize=9.5)
    save(fig, "f14_ruido")


def f_passos():
    mps = [2, 3, 4, 6, 10]; M, L, H, T = [], [], [], []
    for mp in mps:
        r = avaliar(TAREFAS, ruido(0.4), KB, K=K, max_passos=mp)
        m, lo, hi = av.bootstrap_ic(r["taxa"], n=1000, seed=0); M.append(m); L.append(lo); H.append(hi); T.append(r["tokens"].mean())
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    ax.errorbar(mps, M, yerr=[np.array(M) - np.array(L), np.array(H) - np.array(M)], fmt="o-", c=IND, capsize=3, label="sucesso (IC 95%)")
    ax.set_xlabel("limite de passos"); ax.set_ylabel("taxa de sucesso"); ax.set_ylim(0, 1)
    b = ax.twinx(); b.plot(mps, T, "s--", c=GRN, label="tokens (proxy)"); b.set_ylabel("tokens de entrada (proxy)", color=GRN); b.spines["right"].set_visible(True)
    ax.set_title("Limite de passos: sucesso e custo (p = 0,4)", fontsize=9.5); save(fig, "f14_passos")


def f_injecao():
    inj = [t for t in TAREFAS if t.tipo == "injecao"]; sim = lambda n, a: True; nao = lambda n, a: False
    abre = dict(dominios_email=("",))
    cfg = [("nenhuma", dict(sanitizar=False, politica=Politica(**abre), confirmador=sim)),
           ("sanitizador", dict(sanitizar=True, politica=Politica(**abre), confirmador=sim)),
           ("domínio\npermitido", dict(sanitizar=False, politica=Politica(), confirmador=sim)),
           ("menor\nprivilégio", dict(sanitizar=False, politica=Politica(permitidas={"buscar_base"}, **abre), confirmador=sim)),
           ("confirmação\nhumana", dict(sanitizar=False, politica=Politica(**abre), confirmador=nao))]
    ind = []; ok = []
    for nome, kw in cfg:
        r = avaliar(inj, Falhas(susceptibilidade=1.0), KB, K=K, **kw)
        ind.append(r["desfechos"]["ação indevida"] / (len(inj) * K) * 100); ok.append(r["desfechos"]["sucesso"] / (len(inj) * K) * 100)
    x = np.arange(len(cfg)); fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(x - 0.2, ind, 0.4, color=RED, label="ação indevida executada (%)"); ax.bar(x + 0.2, ok, 0.4, color=GRN, label="tarefa concluída corretamente (%)")
    ax.set_xticks(x); ax.set_xticklabels([c[0] for c in cfg], fontsize=8); ax.set_ylabel("% das 100 execuções"); ax.set_ylim(0, 115)
    ax.set_title("Injeção indireta (modelo 100% obediente): efeito de cada defesa", fontsize=9.2); ax.legend(fontsize=7.5, frameon=False, loc="upper center", ncol=2); save(fig, "f14_injecao")


def f_tipos():
    r = avaliar(TAREFAS, ruido(0.2), KB, K=K, max_passos=6); tipos = np.array(r["tipos"]); ordem = list(dict.fromkeys(tipos))
    ag = [r["taxa"][tipos == t].mean() for t in ordem]; wf = []
    for t in ordem:
        oks = []
        for tk in [x for x in TAREFAS if x.tipo == t]:
            reg = criar_registro(Ambiente(kb=KB, hoje=tk.hoje)); resp, _ = fluxo_roteamento(tk.texto, reg)
            oks.append(all(e.lower() in resp.lower() for e in tk.esperado))
        wf.append(np.mean(oks))
    x = np.arange(len(ordem)); fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(x - 0.2, wf, 0.4, color=GRN, label="fluxo com roteamento (código)"); ax.bar(x + 0.2, ag, 0.4, color=IND, label="agente (falhas p = 0,2)")
    ax.set_xticks(x); ax.set_xticklabels([o.replace("calculo", "cálculo").replace("injecao", "injeção") for o in ordem], fontsize=8); ax.set_ylabel("taxa de sucesso"); ax.set_ylim(0, 1.25); ax.legend(fontsize=7.5, frameon=False, ncol=2, loc="upper center")
    ax.set_title("Fluxo fixo x agente, por tipo de tarefa", fontsize=9.5); save(fig, "f14_tipos")


def f_falhas():
    ps = [0, 0.4, 0.8]; cats = ["sucesso", "resposta incorreta", "laco", "max_passos", "orcamento", "ação indevida"]; cores = [GRN, AMB, RED, VIO, PNK, NAVY]
    dados = []
    for p in ps:
        r = avaliar(TAREFAS, ruido(p), KB, K=K, max_passos=6); n = sum(r["desfechos"].values()); dados.append([r["desfechos"][c] / n * 100 for c in cats])
    dados = np.array(dados); fig, ax = plt.subplots(figsize=(6.2, 3.4)); base = np.zeros(len(ps))
    for j, c in enumerate(cats):
        ax.bar([str(p) for p in ps], dados[:, j], bottom=base, color=cores[j], label=c.replace("laco", "laço").replace("max_passos", "limite de passos").replace("orcamento", "orçamento")); base += dados[:, j]
    ax.set_xlabel("intensidade p das falhas injetadas"); ax.set_ylabel("% das execuções"); ax.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_title("Desfechos das execuções", fontsize=9.5); save(fig, "f14_falhas")


if __name__ == "__main__":
    mapa(); chamada(); ciclo(); fluxos(); camadas(); f_ruido(); f_passos(); f_injecao(); f_tipos(); f_falhas()
