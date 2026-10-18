"""Figuras do Capítulo 8. f08_resultados é baseada em dados do SIMULADOR de regras (não de um LLM); as demais são esquemas."""
import sys; sys.path.insert(0, "/home/user/claude/llm/build"); sys.path.insert(0, "/home/user/claude/llm/codigo/comum")
import re, json
from fc import *
from llmdidatico import corpus, avaliacao as av
from llmdidatico.simulador import SimuladorLLM, INTENCOES


def mapa():
    fig, ax = canvas(9.4, 4.2)
    rbox(ax, 2, 62, 28, 28, "Componentes do prompt\npapel · tarefa · contexto\nexemplos · formato", fc="#E0E7FF", ec=IND, fs=8.6, bold=True)
    rbox(ax, 36, 62, 28, 28, "Estrutura e\ndecomposição\ndelimitadores · etapas", fc="#CFFAFE", ec=CYA, fs=8.6, bold=True)
    rbox(ax, 70, 62, 28, 28, "Saída\nvalidável\nJSON · esquema", fc="#D1FAE5", ec=GRN, fs=8.8, bold=True)
    arrow(ax, (30.5, 76), (35.5, 76), c=INK); arrow(ax, (64.5, 76), (69.5, 76), c=INK)
    rbox(ax, 2, 10, 44, 34, "Justificativas verificáveis\ncitação literal · cálculo explícito\nevidência · abstenção", fc="#FEF3C7", ec=AMB, fs=8.6, bold=True)
    rbox(ax, 54, 10, 44, 34, "Avaliação e versionamento\nconjunto de testes · métricas · IC\nregistro de modelo e parâmetros", fc="#FCE7F3", ec=PNK, fs=8.6, bold=True)
    arrow(ax, (84, 61.5), (84, 45), c=GRY, ls="--"); arrow(ax, (50, 61.5), (30, 45), c=GRY, ls="--"); arrow(ax, (46.5, 27), (53.5, 27), c=INK)
    save(fig, "f08_mapa")


def anatomia():
    fig, ax = canvas(9.0, 4.6, (0, 100), (0, 62))
    partes = [("Papel / sistema", "Você é um assistente de triagem…", IND), ("Tarefa", "Classifique a mensagem em UMA categoria…", CYA), ("Contexto", "categorias permitidas; política; documentos", GRN), ("Exemplos", "mensagem → resposta esperada", AMB), ("Formato de saída", "Responda apenas com JSON {…}", PNK), ("Entrada", "Mensagem: <<<texto do cliente>>>", VIO)]
    for k, (t, d, c) in enumerate(partes):
        y = 52 - k * 9.2; rbox(ax, 2, y, 22, 7.6, t, fc=c, ec=c, tc="white", fs=8.2, bold=True); rbox(ax, 26, y, 72, 7.6, d, fc="white", ec=c, fs=8.2, ha="left")
    save(fig, "f08_anatomia")


def verif():
    fig, ax = canvas(9.2, 3.8)
    rbox(ax, 2, 56, 24, 30, "Resposta do\nmodelo\n+ artefatos", fc=IND, ec=IND, tc="white", fs=9, bold=True)
    itens = [("citação literal", "busca no documento", GRN), ("cálculo explícito", "refazer em código", AMB), ("campos estruturados", "validar o esquema", CYA), ("abstenção", "testar sem resposta", PNK)]
    for k, (a, b, c) in enumerate(itens):
        y = 78 - k * 20; rbox(ax, 36, y - 6, 28, 14, a, fc="white", ec=c, fs=8.6, bold=True); rbox(ax, 70, y - 6, 28, 14, b, fc=c, ec=c, tc="white", fs=8.4, bold=True)
        arrow(ax, (26.4, 71), (35.6, y + 1), c=GRY, lw=1.2, ms=9); arrow(ax, (64.4, y + 1), (69.6, y + 1), c=INK, lw=1.2, ms=9)
    ax.text(50, 2, "verifica-se o artefato, e não a \"explicação\"", ha="center", fontsize=8.8, color=GRY, style="italic")
    save(fig, "f08_verif")


def resultados():
    def validar(txt):
        try: o = json.loads(txt); return o.get("intencao") in INTENCOES + ["indefinido"], o.get("intencao")
        except Exception: return False, None
    def interpretar(v, t):
        if v == "v3": ok, r = validar(t); return (r, True) if ok else (None, False)
        if v == "v2": m = re.fullmatch(r"rotulo: (\w+)", t.strip()); return (m.group(1), True) if m else (None, False)
        a = [c for c in INTENCOES if c.replace("_", " ") in t]; return (a[0], False) if a else (None, False)
    X, y = corpus.intencoes(60, seed=11); res = {}
    for v in ("v1", "v2", "v3"):
        sim = SimuladorLLM(v, semente=0); ac = [float(interpretar(v, sim.responder(t))[0] == r) for t, r in zip(X, y)]; res[v] = av.bootstrap_ic(ac, n=1000, seed=0)
    fig, ax = plt.subplots(figsize=(5.8, 3.6)); nomes = ["v1\nvago", "v2\ncategorias e formato", "v3\nexemplos e JSON"]; cores = [GRY, CYA, IND]
    for i, (v, c) in enumerate(zip(("v1", "v2", "v3"), cores)):
        m, lo, hi = res[v]; ax.bar(i, m, color=c, width=0.55); ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], c=INK, capsize=5); ax.text(i, hi + 0.02, f"{m:.2f}", ha="center", fontsize=9)
    ax.set_xticks(range(3)); ax.set_xticklabels(nomes, fontsize=8.5); ax.set_ylim(0, 1.1); ax.set_ylabel("acerto (IC 95% por bootstrap)"); ax.set_title("Três versões de prompt (SIMULADOR, não LLM)", fontsize=9.5)
    save(fig, "f08_resultados")


if __name__ == "__main__":
    mapa(); anatomia(); verif(); resultados()
