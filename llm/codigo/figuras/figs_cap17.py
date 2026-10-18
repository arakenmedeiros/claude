"""Figuras do Capítulo 17. As figuras f17_asr, f17_pii, f17_equidade e f17_custo são baseadas em dados calculados pelos blocos de código
do próprio capítulo (cap/ch17.md), executados aqui em sequência: ASSISTENTE SIMULADO (regras, NÃO é um LLM), textos e dados
SINTÉTICOS. As demais são esquemas conceituais."""
import sys; sys.path.insert(0, "/home/user/claude/llm/build"); sys.path.insert(0, "/home/user/claude/llm/codigo/comum")
import re, io, contextlib
from fc import *


def dados_do_capitulo():
    """Executa os blocos ```python / ```python+ de ch17.md numa única sequência (sem a demonstração do pickle) e devolve o espaço de nomes."""
    s = open("/home/user/claude/llm/cap/ch17.md").read()
    blocos = re.findall(r"^```python\+?\n(.*?)^```", s, re.S | re.M)
    ns = {"__name__": "figs_cap17_exec"}
    with contextlib.redirect_stdout(io.StringIO()):
        for b in blocos:
            if "pickle" in b:
                b = b.split("import pickle")[0]
            exec(compile(b, "ch17_bloco", "exec"), ns)
    return ns


def mapa():
    fig, ax = canvas(9.6, 4.6)
    topo = [("Riscos\n8 famílias:\nalucinação, viés,\nprivacidade,\nsegurança…", IND, "#E0E7FF"),
            ("Modelo de\nameaças\nSTRIDE adaptado", CYA, "#CFFAFE"),
            ("Ameaças\ninjeção, vazamento,\nexfiltração, RAG,\ncusto", RED, "#FEE2E2"),
            ("Mitigações\nem camadas\nprivilégio mínimo", GRN, "#D1FAE5"),
            ("Red teaming\ncatálogo, variantes\ntaxa com IC", AMB, "#FEF3C7")]
    for k, (t, c, f) in enumerate(topo):
        x = 1 + k * 20; rbox(ax, x, 56, 17.5, 34, t, fc=f, ec=c, fs=7.2, bold=True)
        if k < 4: arrow(ax, (x + 17.9, 73), (x + 20 - 0.4, 73), c=INK, lw=1.3, ms=10)
    base = [("Dados pessoais\nLGPD, PII,\nk-anonimato", PNK, "#FCE7F3"), ("Equidade e\ntransparência", VIO, "#EDE9FE"),
            ("Licenças e\ndireitos\nautorais", AMB, "#FEF3C7"), ("Regulação\nNIST, AI Act,\nPL 2.338/2023", CYA, "#CFFAFE"),
            ("Governança\ninventário, RACI,\nauditoria", IND, "#E0E7FF")]
    for k, (t, c, f) in enumerate(base):
        x = 1 + k * 20; rbox(ax, x, 8, 17.5, 28, t, fc=f, ec=c, fs=7.2, bold=True)
    arrow(ax, (50, 37), (50, 55), c=GRY, ls="--", lw=1.2, ms=10)
    ax.text(51.5, 46, "controles transversais", fontsize=7.6, color=GRY, style="italic", va="center")
    save(fig, "f17_mapa")


def stride():
    fig, ax = canvas(9.4, 4.6)
    rbox(ax, 0.5, 3, 62, 94, "", fc="#FFF7ED", ec=AMB, lw=1.2, alpha=0.6)
    ax.text(2.5, 93.5, "dentro: o que você controla", fontsize=7.6, color=AMB, style="italic")
    rbox(ax, 1.5, 62, 15, 20, "Usuário", fc="white", ec=INK, fs=8.5, bold=True)
    rbox(ax, 1.5, 22, 15, 20, "Documentos\nBase RAG", fc="white", ec=INK, fs=8.2, bold=True)
    rbox(ax, 26, 38, 18, 26, "Assistente\n(LLM + prompt\nde sistema)", fc="#E0E7FF", ec=IND, fs=8.4, bold=True)
    rbox(ax, 70, 62, 27, 20, "Ferramentas\ne-mail · notas · registros", fc="white", ec=INK, fs=8.2, bold=True)
    rbox(ax, 70, 22, 27, 20, "Saída e logs\nresposta · rastros", fc="white", ec=INK, fs=8.2, bold=True)
    arrow(ax, (16.8, 72), (25.6, 56), c=INK); arrow(ax, (16.8, 32), (25.6, 46), c=INK)
    arrow(ax, (44.4, 56), (69.6, 72), c=INK); arrow(ax, (44.4, 46), (69.6, 32), c=INK)
    ax.plot([63, 63], [3, 97], color=RED, lw=1.2, ls="--"); ax.text(64, 93.5, "fronteira", fontsize=7.2, color=RED)
    tags = [("S", 9, 84, "falsa origem"), ("T", 9, 44, "adulteração"), ("I", 35, 66.5, "vazamento"), ("E", 83, 84, "excesso de poder"), ("R", 83, 44, "sem rastro"), ("D", 35, 34.5, "custo")]
    for L, x, y, txt in tags:
        c = Circle((x, y), 2.4, color=RED, zorder=5); ax.add_patch(c); ax.text(x, y, L, color="white", ha="center", va="center", fontsize=8, fontweight="bold", zorder=6)
        ax.text(x + 3.4, y, txt, fontsize=7.4, va="center", color=INK)
    ax.text(50, -2, "S falsa origem · T adulteração · R ausência de rastro · I vazamento · D indisponibilidade/custo · E excesso de poder", ha="center", fontsize=7.4, color=GRY)
    save(fig, "f17_stride")


def injecao():
    fig, ax = canvas(9.4, 4.4)
    ax.text(1, 95, "Injeção DIRETA", fontsize=9, fontweight="bold", color=RED)
    rbox(ax, 1, 66, 18, 20, "Usuário\nmalicioso", fc="#FEE2E2", ec=RED, fs=8.2, bold=True)
    rbox(ax, 36, 62, 30, 28, "Contexto do modelo\nregras do sistema\n+ mensagem com a ordem", fc="#E0E7FF", ec=IND, fs=8.2, bold=True)
    rbox(ax, 80, 66, 18, 20, "Ação ou\nvazamento", fc="#FEF3C7", ec=AMB, fs=8.2, bold=True)
    arrow(ax, (19.4, 76), (35.6, 76), c=RED); arrow(ax, (66.4, 76), (79.6, 76), c=INK)
    ax.plot([2, 98], [50, 50], color=MID, lw=1)
    ax.text(1, 44, "Injeção INDIRETA", fontsize=9, fontweight="bold", color=RED)
    rbox(ax, 1, 8, 15, 22, "Autor do\nconteúdo", fc="#FEE2E2", ec=RED, fs=8, bold=True)
    rbox(ax, 22, 8, 17, 22, "Página, PDF,\ne-mail ou\ndocumento", fc="white", ec=INK, fs=8, bold=True)
    rbox(ax, 44, 8, 14, 22, "Busca /\nRAG", fc="white", ec=INK, fs=8, bold=True)
    rbox(ax, 63, 8, 18, 22, "Contexto do\nmodelo", fc="#E0E7FF", ec=IND, fs=8, bold=True)
    rbox(ax, 86, 8, 13, 22, "Ação", fc="#FEF3C7", ec=AMB, fs=8.2, bold=True)
    for a, b in [(16.4, 21.6), (39.4, 43.6), (58.4, 62.6), (81.4, 85.6)]:
        arrow(ax, (a, 19), (b, 19), c=INK, lw=1.3, ms=10)
    rbox(ax, 63, 34, 18, 9, "Usuário inocente\npergunta", fc="#D1FAE5", ec=GRN, fs=7.4)
    arrow(ax, (72, 33.6), (72, 30.4), c=GRN, lw=1.2, ms=9)
    save(fig, "f17_injecao")


def camadas():
    fig, ax = canvas(9.4, 4.2)
    nomes = ["1 Delimitadores\ne instruções", "2 Filtro de\nentrada", "3 Filtro de\nsaída", "4 Privilégio\nmínimo", "5 Aprovação\nhumana"]
    xs = [24, 38, 52, 66, 80]; furos = [[40, 62], [40, 76], [40, 56], [40, 24], [40, 70]]
    cores = [CYA, IND, VIO, GRN, AMB]
    for x, nm, fu, c in zip(xs, nomes, furos, cores):
        rbox(ax, x, 18, 7, 70, "", fc=c, ec=c, alpha=0.85, r=1)
        for y in fu: ax.add_patch(Ellipse((x + 3.5, y), 6.4, 9, color="white", zorder=4))
        ax.text(x + 3.5, 8, nm, ha="center", va="center", fontsize=7.4)
    rbox(ax, 0.5, 44, 12, 20, "Ataque", fc="#FEE2E2", ec=RED, fs=8.5, bold=True)
    rbox(ax, 90.5, 44, 9, 20, "Dano", fc="#FEF3C7", ec=AMB, fs=8.5, bold=True)
    ax.plot([13, 25], [54, 54], color=RED, lw=1.8); ax.text(30.5, 54, "×", color=RED, fontsize=15, ha="center", va="center", zorder=6)
    ax.plot([13, 39], [62, 62], color=RED, lw=1.8); ax.text(41.5, 62, "×", color=RED, fontsize=15, ha="center", va="center", zorder=6)
    arrow(ax, (13, 40), (90, 40), c=RED, lw=1.8, ms=12)
    ax.text(51, 94, "a maioria é barrada; algum ataque pode encontrar furos alinhados", ha="center", fontsize=8, style="italic", color=GRY)
    ax.text(95, 33, "o que sobra:\nlimitar o dano", ha="center", fontsize=7.4, color=RED)
    save(fig, "f17_camadas")


def governanca():
    fig, ax = canvas(8.6, 4.6, (0, 100), (0, 100))
    etapas = ["Inventariar\nsistemas", "Classificar\no risco", "Aprovar com\nevidências", "Operar com\nsupervisão humana", "Monitorar\npós-implantação", "Registrar\nincidentes", "Auditar e\nreavaliar"]
    cores = [IND, CYA, GRN, AMB, PNK, RED, VIO]
    cx, cy, rx, ry = 50, 50, 36, 34; pos = []
    for k, (t, c) in enumerate(zip(etapas, cores)):
        ang = np.pi / 2 - 2 * np.pi * k / len(etapas); x = cx + rx * np.cos(ang); y = cy + ry * np.sin(ang); pos.append((x, y))
        rbox(ax, x - 10.5, y - 7, 21, 14, t, fc="white", ec=c, fs=7.8, bold=True, lw=1.8)
    for k in range(len(etapas)):
        (x1, y1), (x2, y2) = pos[k], pos[(k + 1) % len(etapas)]
        dx, dy = x2 - x1, y2 - y1; n = np.hypot(dx, dy); u = (dx / n, dy / n)
        arrow(ax, (x1 + u[0] * 12, y1 + u[1] * 8.5), (x2 - u[0] * 12, y2 - u[1] * 8.5), c=GRY, lw=1.3, ms=10)
    ax.text(cx, cy, "Governança\n(NIST: Govern)", ha="center", va="center", fontsize=9, fontweight="bold", color=INK)
    save(fig, "f17_governanca")


def asr(ns):
    tab = ns["tab_camadas"]; nomes = ["0\nnenhuma", "1\n+delim.", "2\n+entrada", "3\n+saída", "4\n+privilégio", "5\n+aprovação"]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for i, (_, m, lo, hi, uso) in enumerate(tab):
        ax.bar(i, m, color=IND, width=0.6); ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], c=INK, capsize=4); ax.text(i, hi + 0.02, f"{m:.2f}", ha="center", fontsize=8.5)
    ax.set_xticks(range(6)); ax.set_xticklabels(nomes, fontsize=8); ax.set_ylim(0, 0.8); ax.set_ylabel("taxa de sucesso do ataque (IC 95%)")
    ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
    ax2.plot(range(6), [t[4] for t in tab], color=AMB, marker="o", lw=2); ax2.set_ylim(0, 1.15); ax2.set_ylabel("mensagens legítimas respondidas", color=AMB)
    ax.set_title("Camadas somadas (SIMULAÇÃO, não LLM)", fontsize=9.5)
    save(fig, "f17_asr")


def pii(ns):
    r = ns["resumo_pii"]; tipos = ["CPF", "EMAIL", "TEL", "CEP", "NASC", "PRONT", "NOME"]
    def rec(tot, t): v = tot.get(t, [0, 0, 0]); return v[0] / (v[0] + v[2]) if v[0] + v[2] else np.nan
    def pre(tot, t): v = tot.get(t, [0, 0, 0]); return v[0] / (v[0] + v[1]) if v[0] + v[1] else np.nan
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.6), sharey=True); x = np.arange(len(tipos)); w = 0.38
    for ax, f, tit in [(axs[0], rec, "Revocação"), (axs[1], pre, "Precisão")]:
        ax.bar(x - w / 2, [f(r["fácil"][0], t) for t in tipos], w, color=CYA, label="conjunto fácil")
        ax.bar(x + w / 2, [f(r["difícil"][0], t) for t in tipos], w, color=RED, label="conjunto difícil")
        ax.set_xticks(x); ax.set_xticklabels(tipos, rotation=40, fontsize=8); ax.set_title(tit, fontsize=9.5); ax.set_ylim(0, 1.08)
    axs[0].legend(fontsize=8, loc="lower left"); axs[0].set_ylabel("por tipo de dado")
    save(fig, "f17_pii")


def equidade(ns):
    tab = ns["tab_eq"]; nomes = list(tab); rot = ["limiar\núnico", "paridade\ndemográfica", "igual\noportunidade"]
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.5), sharey=True); x = np.arange(3); w = 0.36
    for ax, chave, tit in [(axs[0], "sel", "Taxa de seleção"), (axs[1], "tpr", "Taxa de verdadeiros positivos")]:
        a = [tab[n][0]["A"][chave] for n in nomes]; b = [tab[n][0]["B"][chave] for n in nomes]
        ax.bar(x - w / 2, a, w, color=IND, label="grupo A"); ax.bar(x + w / 2, b, w, color=AMB, label="grupo B")
        for i in range(3): ax.text(i - w / 2, a[i] + 0.01, f"{a[i]:.2f}", ha="center", fontsize=7.6); ax.text(i + w / 2, b[i] + 0.01, f"{b[i]:.2f}", ha="center", fontsize=7.6)
        ax.set_xticks(x); ax.set_xticklabels(rot, fontsize=8); ax.set_title(tit, fontsize=9.5); ax.set_ylim(0, 0.95)
    axs[0].legend(fontsize=8, loc="upper right")
    save(fig, "f17_equidade")


def custo(ns):
    from llmdidatico.seguranca import simular_custo
    ms = range(1, 11)
    sem = [simular_custo(minutos=m)[0]["atacante"] for m in ms]
    com = [simular_custo(minutos=m, limite_req_min=5, max_tokens_req=1000, orcamento_tokens=50_000)[0]["atacante"] for m in ms]
    leg = [simular_custo(minutos=m)[0]["legitimo"] for m in ms]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(ms, sem, color=RED, marker="o", lw=2, label="atacante, sem limites"); ax.plot(ms, com, color=GRN, marker="s", lw=2, label="atacante, com limites")
    ax.plot(ms, leg, color=IND, marker="^", lw=2, label="20 usuários legítimos"); ax.set_yscale("log")
    ax.set_xlabel("minutos"); ax.set_ylabel("custo acumulado (unidades de token, fictícias)"); ax.legend(fontsize=8); ax.set_title("Negação de serviço por custo (SIMULAÇÃO)", fontsize=9.5)
    save(fig, "f17_custo")


if __name__ == "__main__":
    mapa(); stride(); injecao(); camadas(); governanca()
    ns = dados_do_capitulo()
    asr(ns); pii(ns); equidade(ns); custo(ns)
