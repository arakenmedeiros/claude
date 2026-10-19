"""Figuras do Capítulo 18. As figuras f18_p1_limiar, f18_p2_guarda, f18_p3_modos, f18_p4_custo, f18_p5_extracao e f18_p6_bancada são
baseadas em dados calculados pelos blocos de código do próprio capítulo (cap/ch18.md), executados aqui em sequência: SIMULADORES de regras
(NÃO são LLMs), dados SINTÉTICOS e FICTÍCIOS, preços e latências HIPOTÉTICOS. As demais são esquemas conceituais."""
import sys; sys.path.insert(0, "/home/user/claude/llm/build"); sys.path.insert(0, "/home/user/claude/llm/codigo/comum")
import re, io, contextlib
from fc import *


def dados_do_capitulo():
    s = open("/home/user/claude/llm/cap/ch18.md").read()
    blocos = re.findall(r"^```python\+?\n(.*?)^```", s, re.S | re.M)
    ns = {"__name__": "figs_cap18_exec"}
    with contextlib.redirect_stdout(io.StringIO()):
        for b in blocos:
            exec(compile(b, "ch18_bloco", "exec"), ns)
    return ns


def fluxo(name, caixas, w=9.6, h=3.4, linhas=None, notas=None, titulo=None):
    """caixas: lista de (texto, cor, fundo) em uma linha; linhas: segunda linha opcional (lista igual) ligada à última caixa da primeira."""
    fig, ax = canvas(w, h)
    n = len(caixas); lw = 96 / n; bw = lw - 3.2
    y1 = 58 if linhas else 38
    if titulo: ax.text(1, 96, titulo, fontsize=8.6, fontweight="bold", color=INK)
    for k, (t, c, f) in enumerate(caixas):
        x = 2 + k * lw; rbox(ax, x, y1, bw, 32, t, fc=f, ec=c, fs=6.2, bold=True)
        if k < n - 1: arrow(ax, (x + bw + 0.3, y1 + 16), (x + lw - 0.3, y1 + 16), c=INK, lw=1.3, ms=10)
    if linhas:
        m = len(linhas); lw2 = 96 / m; bw2 = lw2 - 4.5
        for k, (t, c, f) in enumerate(linhas):
            x = 2 + k * lw2; rbox(ax, x, 8, bw2, 28, t, fc=f, ec=c, fs=6.4, bold=True)
            if k < m - 1: arrow(ax, (x + bw2 + 0.3, 22), (x + lw2 - 0.3, 22), c=INK, lw=1.3, ms=10)
        arrow(ax, (2 + (n - 1) * lw + bw / 2, y1 - 0.5), (2 + bw2 / 2 + 2, 36.5), c=GRY, ls="--", lw=1.1, ms=9, rad=-0.15)
    if notas:
        ax.text(50, 1.5, notas, ha="center", fontsize=7.2, color=GRY, style="italic")
    save(fig, name)


def mapa():
    fig, ax = canvas(9.6, 4.8)
    rbox(ax, 30, 40, 40, 20, "O molde\nproblema · requisitos · dados · arquitetura\navaliação com IC · erros · riscos · relatório", fc="#E0E7FF", ec=IND, fs=7.8, bold=True)
    projs = [("P1 Educação\nRAG de normas", IND, "#E0E7FF"), ("P2 Saúde\ntriagem informativa", PNK, "#FCE7F3"), ("P3 Jurídico\nvigência e perfis", AMB, "#FEF3C7"),
             ("P4 Atendimento\nagente e custo", GRN, "#D1FAE5"), ("P5 Documentação\nextração validada", CYA, "#CFFAFE"), ("P6 Integrador\nbancada e governança", RED, "#FEE2E2")]
    pos = [(3, 70), (36, 78), (69, 70), (3, 8), (36, 2), (69, 8)]
    for (t, c, f), (x, y) in zip(projs, pos):
        rbox(ax, x, y, 28, 18, t, fc=f, ec=c, fs=7.6, bold=True)
        arrow(ax, (x + 14, y + (0 if y > 50 else 18)), (50, 60 if y > 50 else 40), c=GRY, lw=1.1, ms=9, ls="--")
    rbox(ax, 70, 44, 28, 12, "equipe · repositório\nreprodutibilidade · limites", fc="white", ec=GRY, fs=7.2)
    rbox(ax, 2, 44, 26, 12, "camadas dos\ncapítulos 9 a 17", fc="white", ec=GRY, fs=7.2)
    save(fig, "f18_mapa")


def molde():
    et = [("Problema e\nusuários", IND, "#E0E7FF"), ("Requisitos +\ncritérios", AMB, "#FEF3C7"), ("Dados\n(origem)", CYA, "#CFFAFE"), ("Arquitetura", GRN, "#D1FAE5"),
          ("Implementação\nexecutada", VIO, "#EDE9FE"), ("Avaliação\ncom IC", RED, "#FEE2E2"), ("Erros e\nriscos", PNK, "#FCE7F3"), ("Relatório", INK, MID)]
    fig, ax = canvas(9.8, 3.4)
    lw = 97 / len(et); bw = lw - 2.6
    for k, (t, c, f) in enumerate(et):
        x = 1 + k * lw; rbox(ax, x, 40, bw, 28, t, fc=f, ec=c, fs=6.8, bold=True)
        if k < len(et) - 1: arrow(ax, (x + bw + 0.2, 54), (x + lw - 0.2, 54), c=INK, lw=1.2, ms=9)
    arrow(ax, (1 + 6.5 * lw + bw / 2, 39), (1 + 2.5 * lw + bw / 2, 39), c=GRY, rad=-0.4, ls="--", lw=1.2, ms=10)
    ax.text(50, 8, "a análise de erros realimenta requisitos, dados e arquitetura", ha="center", fontsize=7.6, color=GRY, style="italic")
    ax.text(1 + 1.5 * lw, 73, "escrito ANTES de medir", fontsize=7.2, color=AMB, ha="center", fontweight="bold")
    save(fig, "f18_molde")


def repo():
    fig, ax = canvas(9.2, 3.8)
    itens = [("dados/", "conjuntos +\nREADME (origem,\nlicença, limites)", CYA, "#CFFAFE"), ("src/", "pacote do projeto\nmódulos pequenos", IND, "#E0E7FF"), ("configs/", "um arquivo por\nexperimento", AMB, "#FEF3C7"),
             ("bancada/", "casos por estrato\n+ critérios", GRN, "#D1FAE5"), ("resultados/", "saídas com data,\nhash de config\ne dos casos", VIO, "#EDE9FE"), ("docs/", "requisitos,\ncartão, governança,\nrelatórios", RED, "#FEE2E2")]
    rbox(ax, 36, 78, 28, 16, "projeto/", fc=NAVY, ec=NAVY, tc="white", fs=9, bold=True)
    for k, (n, d, c, f) in enumerate(itens):
        x = 1 + k * 16.4
        rbox(ax, x, 12, 15, 40, f"{n}\n\n{d}", fc=f, ec=c, fs=6.0, bold=True)
        arrow(ax, (50, 77.6), (x + 7.5, 52.4), c=GRY, lw=1.0, ms=8)
    ax.text(50, 3, "cada resultado aponta para o código, a configuração e os dados que o geraram", ha="center", fontsize=7.2, color=GRY, style="italic")
    save(fig, "f18_repo")


def arqs():
    fluxo("f18_p1_arq", [("Pergunta", INK, LIGHT), ("Busca híbrida\nBM25 + LSA", IND, "#E0E7FF"), ("Prompt com\ndelimitadores", CYA, "#CFFAFE"), ("Gerador\nSIMULADO", AMB, "#FEF3C7"),
                         ("Verificação de\ncitação", GRN, "#D1FAE5"), ("Resposta\nou abstenção", RED, "#FEE2E2")],
          notas="cada etapa é medida separadamente: recuperação, geração, fidelidade, abstenção")
    fluxo("f18_p2_arq", [("Pergunta", INK, LIGHT), ("Mascarar\ndados pessoais", PNK, "#FCE7F3"), ("Guardrails\nurgência / clínica", RED, "#FEE2E2"), ("RAG de serviços\n(só serviço)", IND, "#E0E7FF"),
                         ("Resposta + rodapé\nou abstenção", GRN, "#D1FAE5")],
          linhas=[("Texto FIXO de\nurgência (192)", RED, "#FEE2E2"), ("Texto FIXO:\nprocure profissional", AMB, "#FEF3C7"), ("Registro só com\ntexto mascarado", PNK, "#FCE7F3")],
          notas="regras determinísticas e imperfeitas: medidas em conjunto cego")
    fluxo("f18_p3_arq", [("Pergunta +\ndata + perfil", INK, LIGHT), ("Filtro de vigência\ne de perfil (PRÉ)", RED, "#FEE2E2"), ("Busca híbrida\nnas normas", IND, "#E0E7FF"), ("Gerador SIMULADO\n+ verificação", AMB, "#FEF3C7"),
                         ("Resposta com\nidentificação da norma", GRN, "#D1FAE5")],
          linhas=[("Usuário\npseudonimizado", PNK, "#FCE7F3"), ("Trilha de auditoria\nencadeada por hash", VIO, "#EDE9FE"), ("Verificação de\nintegridade", CYA, "#CFFAFE")],
          notas="o filtro vem ANTES de escolher os k melhores")
    fluxo("f18_p4_arq", [("Mensagem", INK, LIGHT), ("Classificador\nde intenção", IND, "#E0E7FF"), ("Automatizável?", AMB, "#FEF3C7"), ("Agente + ferramentas\n(menor privilégio)", GRN, "#D1FAE5"),
                         ("Resposta", CYA, "#CFFAFE")],
          linhas=[("Não: atendente\nhumano (sem custo\nde modelo)", RED, "#FEE2E2"), ("Ação irreversível:\nconfirmação", VIO, "#EDE9FE"), ("Plano B: repetir\ne transferir", PNK, "#FCE7F3")],
          notas="preços e latências do exemplo são HIPOTÉTICOS")
    fluxo("f18_p5_arq", [("Página de\ndocumentação", INK, LIGHT), ("Busca BM25", IND, "#E0E7FF"), ("Extrator\n(JSON como texto)", AMB, "#FEF3C7"), ("Reparo do\nformato", CYA, "#CFFAFE"), ("Esquema\npydantic", GRN, "#D1FAE5"),
                         ("JSON válido\nou revisão humana", RED, "#FEE2E2")],
          notas="válido não é o mesmo que correto: confira com gabarito ou revisão por amostragem")
    fluxo("f18_p6_arq", [("Pedido", INK, LIGHT), ("Entrada: PII,\nfiltro, custo", PNK, "#FCE7F3"), ("Roteamento", AMB, "#FEF3C7"), ("RAG  ou  agente\n(camadas 11 e 14)", IND, "#E0E7FF"), ("Provedor com\nrobustez (cap. 9)", CYA, "#CFFAFE"),
                         ("Validação e\nfiltro de saída", GRN, "#D1FAE5")],
          linhas=[("Auditoria\nencadeada", VIO, "#EDE9FE"), ("Bancada por estrato\n+ critérios (cap. 16)", RED, "#FEE2E2"), ("Relatório automático\ncartão · governança", INK, MID)],
          notas="a bancada roda fora do caminho do usuário e é refeita a cada mudança")


def p1_limiar(ns):
    v = ns["varredura"]; x = [t[0] for t in v]
    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    for idx, nome, c, mk in [(1, "resposta correta (46 consultas com resposta)", IND, "o"), (2, "abstenção correta (10 sem resposta)", GRN, "s")]:
        y = np.array([t[idx][0] for t in v]); lo = np.array([t[idx][1] for t in v]); hi = np.array([t[idx][2] for t in v])
        ax.errorbar(np.array(x) + (0.004 if idx == 2 else -0.004), y, yerr=[np.maximum(y - lo, 0), np.maximum(hi - y, 0)], color=c, marker=mk, lw=2, capsize=3, label=nome)
    ax.set_xlabel("limiar de abstenção do gerador simulado"); ax.set_ylabel("taxa (IC95% de Wilson)"); ax.set_ylim(0, 1.05); ax.legend(fontsize=7.5, loc="lower center")
    ax.set_title("Projeto 1: o limiar troca um erro por outro (SIMULAÇÃO)", fontsize=9.5)
    save(fig, "f18_p1_limiar")


def p2_guarda(ns):
    P = ns["P"]; res, cego = ns["res"], ns["res_cego"]; guarda, esp = ns["guarda"], ns["esperado_guarda"]
    nomes = [("urgência\n(revocação)", "urgencia"), ("clínica\n(revocação)", "encaminhar_profissional"), ("serviço não\ndesviado", "servico")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.6), gridspec_kw={"width_ratios": [1.7, 1]})
    for off, (conj, nm, c) in zip((-0.19, 0.19), [(res, "desenvolvimento", IND), (cego, "conjunto cego", RED)]):
        ps, los, his = [], [], []
        for _, alvo in nomes:
            t = P.prop_ic([int(guarda(r) == alvo) for q, r in conj if esp(q) == alvo]); ps.append(t[0]); los.append(t[1]); his.append(t[2])
        ps, los, his = map(np.array, (ps, los, his))
        a1.bar(np.arange(3) + off, ps, 0.36, color=c, alpha=0.85, label=nm, yerr=[np.maximum(ps - los, 0), np.maximum(his - ps, 0)], capsize=3, ecolor=INK)
    a1.set_xticks(range(3)); a1.set_xticklabels([n for n, _ in nomes], fontsize=7.8); a1.set_ylim(0, 1.08); a1.set_ylabel("taxa (IC95% de Wilson)")
    a1.axhline(0.99, color=GRY, ls="--", lw=1); a1.text(2.45, 1.0, "critério de urgência", fontsize=6.5, color=GRY, ha="right"); a1.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    a1.set_title("Guardrails: onde as regras foram escritas x cego", fontsize=8.6)
    g = ns["acerto_registro"]; ps, los, his = [], [], []
    for k in ("formal", "coloquial"):
        t = P.prop_ic(g[k]); ps.append(t[0]); los.append(t[1]); his.append(t[2])
    ps, los, his = map(np.array, (ps, los, his))
    a2.bar([0, 1], ps, 0.55, color=[IND, AMB], yerr=[np.maximum(ps - los, 0), np.maximum(his - ps, 0)], capsize=3, ecolor=INK)
    a2.set_xticks([0, 1]); a2.set_xticklabels([f"formal\n(n={len(g['formal'])})", f"coloquial\n(n={len(g['coloquial'])})"], fontsize=7.8); a2.set_ylim(0, 1.05)
    a2.set_title("Serviço respondido\ncorretamente, por registro", fontsize=8.6)
    save(fig, "f18_p2_guarda")


def p3_modos(ns):
    tab = ns["tab3"]; combos = [(m, k) for m in ("ingenuo", "pos_filtro", "pre_filtro") for k in (1, 3)]
    rot = {"ingenuo": "ingênuo", "pos_filtro": "pós-filtro", "pre_filtro": "pré-filtro"}
    fig, ax = plt.subplots(figsize=(6.6, 3.7)); x = np.arange(len(combos)); w = 0.26
    ok = [sum(l["ok"] for l in tab[c]) for c in combos]; vz = [sum(l["vazamento"] for l in tab[c]) for c in combos]; fv = [sum(l["fora_da_vigencia"] for l in tab[c]) for c in combos]
    ax.bar(x - w, ok, w, color=GRN, label="respostas corretas (de 30)"); ax.bar(x, vz, w, color=RED, label="consultas com vazamento de perfil"); ax.bar(x + w, fv, w, color=AMB, label="consultas com norma fora da vigência")
    for xi, v in zip(x - w, ok): ax.text(xi, v + 0.4, str(v), ha="center", fontsize=7)
    for xi, v in zip(x, vz): ax.text(xi, v + 0.4, str(v), ha="center", fontsize=7)
    for xi, v in zip(x + w, fv): ax.text(xi, v + 0.4, str(v), ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([f"{rot[m]}\nk={k}" for m, k in combos], fontsize=7.8); ax.set_ylim(0, 36); ax.legend(fontsize=7.4, loc="upper left")
    ax.set_title("Projeto 3: onde aplicar o filtro de vigência e perfil (SIMULAÇÃO)", fontsize=9.3)
    save(fig, "f18_p3_modos")


def p4_custo(ns):
    from llmdidatico.avaliacao import bootstrap_ic
    res4 = ns["res4"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.6), gridspec_kw={"width_ratios": [1.5, 1]})
    chaves = [(False, 0.0), (True, 0.0), (False, 0.3), (True, 0.3)]
    for i, k in enumerate(chaves):
        pc = np.mean([[r["ok"] for r in run] for run in res4[k]], axis=0); m, lo, hi = bootstrap_ic(pc, n=1000, seed=0)
        a1.bar(i, m, 0.6, color=GRN if k[0] else GRY, yerr=[[max(m - lo, 0)], [max(hi - m, 0)]], capsize=3, ecolor=INK)
    a1.set_xticks(range(4)); a1.set_xticklabels(["sem plano B\nfalha 0", "com plano B\nfalha 0", "sem plano B\nfalha 0,3", "com plano B\nfalha 0,3"], fontsize=7.4)
    a1.set_ylim(0.7, 1.03); a1.set_ylabel("sucesso por caso (IC95% bootstrap)"); a1.set_title("Plano B e falhas de ferramenta", fontsize=8.8)
    lab = ["A: agente\nem tudo", "B: classificador\nantes"]; cu = [ns["custo_A"] * 1e6, ns["custo_B"] * 1e6]; la = [ns["lat_A"], ns["lat_B"]]
    xx = np.arange(2); a2.bar(xx - 0.2, cu, 0.38, color=IND, label="custo (un. hipotéticas por milhão de conversas)")
    a2.set_ylabel("custo por milhão de conversas", fontsize=7.6); a2.set_xticks(xx); a2.set_xticklabels(lab, fontsize=7.6)
    b2 = a2.twinx(); b2.bar(xx + 0.2, la, 0.38, color=AMB); b2.set_ylabel("latência modelada (s)", fontsize=7.6); b2.spines["right"].set_visible(True)
    a2.set_title("Custo (azul) e latência (âmbar)\nHIPOTÉTICOS", fontsize=8.6)
    save(fig, "f18_p4_custo")


def p5_extracao(ns):
    P = ns["P"]; res5 = ns["res5"]
    nomes = [("v1", False), ("v1", True), ("v2", False), ("v2", True)]
    fig, ax = plt.subplots(figsize=(6.4, 3.7)); x = np.arange(4)
    for off, idx, nm, c in [(-0.19, 1, "JSON válido", "#A5B4FC"), (0.19, 2, "todos os campos corretos", IND)]:
        ps, los, his = [], [], []
        for k in nomes:
            t = P.prop_ic(res5[k][idx]); ps.append(t[0]); los.append(t[1]); his.append(t[2])
        ps, los, his = map(np.array, (ps, los, his))
        ax.bar(x + off, ps, 0.36, color=c, label=nm, yerr=[np.maximum(ps - los, 0), np.maximum(his - ps, 0)], capsize=3, ecolor=INK)
    ax.set_xticks(x); ax.set_xticklabels([f"{v}\n{'com' if r else 'sem'} reparo" for v, r in nomes], fontsize=8); ax.set_ylim(0, 1.15)
    ax.axhline(0.95, color=GRY, ls="--", lw=1); ax.text(3.45, 0.965, "critério 0,95", fontsize=6.8, color=GRY, ha="right")
    ax.set_ylabel("taxa (IC95% de Wilson; 36 páginas)"); ax.legend(fontsize=7.6, loc="upper left", ncol=2); ax.set_title("Projeto 5: extração estruturada validada (SIMULAÇÃO)", fontsize=9.3)
    save(fig, "f18_p5_extracao")


def p6_bancada(ns):
    P = ns["P"]; tab = P.tabela_estratos(ns["res_v2"]); v1 = ns["res_v1"]
    fig, ax = plt.subplots(figsize=(6.8, 4.0)); y = np.arange(len(tab))[::-1]
    cores = {"atende": GRN, "não atende": RED, "inconclusivo": AMB}
    for yi, t in zip(y, tab):
        ax.plot([t["lo"], t["hi"]], [yi, yi], color=cores[t["veredito"]], lw=3, solid_capstyle="round")
        ax.plot(t["p"], yi, "o", color=INK, ms=5)
        ax.plot([t["criterio"]] * 2, [yi - 0.32, yi + 0.32], color=NAVY, lw=1.6)
        b = P.prop_ic([r["ok"] for r in v1 if r["estrato"] == t["estrato"]])
        ax.plot(b[0], yi + 0.0, "D", color="white", mec=GRY, ms=5.5, zorder=4)
    ax.set_yticks(y); ax.set_yticklabels([f"{t['estrato']} (n={t['n']})" for t in tab], fontsize=7.8); ax.set_xlim(-0.02, 1.04); ax.set_xlabel("taxa de sucesso (IC95% de Wilson)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], color=GRN, lw=3, label="atende"), Line2D([0], [0], color=AMB, lw=3, label="inconclusivo"), Line2D([0], [0], color=RED, lw=3, label="não atende"),
                       Line2D([0], [0], color=NAVY, lw=1.6, label="critério"), Line2D([0], [0], marker="D", color="w", mec=GRY, label="v1 (sem camadas)")], fontsize=7, loc="lower left")
    ax.set_title("Projeto 6: bancada por estrato (v2, SIMULAÇÃO)", fontsize=9.5)
    save(fig, "f18_p6_bancada")


if __name__ == "__main__":
    mapa(); molde(); repo(); arqs()
    ns = dados_do_capitulo()
    p1_limiar(ns); p2_guarda(ns); p3_modos(ns); p4_custo(ns); p5_extracao(ns); p6_bancada(ns)
