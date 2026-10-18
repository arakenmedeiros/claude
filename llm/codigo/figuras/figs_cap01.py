"""Figuras do Capítulo 1. f01_ppl é gráfico baseado em dados; as demais são esquemas conceituais."""
import sys; sys.path.insert(0, "/home/user/claude/llm/build"); sys.path.insert(0, "/home/user/claude/llm/codigo/comum")
from fc import *
from llmdidatico import corpus, modelos


def mapa():
    fig, ax = canvas(9, 5.2)
    rbox(ax, 2, 72, 22, 20, "Dados\n(texto sem rótulos)", fc="#E0E7FF", ec=IND, fs=10, bold=True)
    rbox(ax, 2, 46, 22, 20, "Parâmetros\n(milhões a bilhões)", fc="#CFFAFE", ec=CYA, fs=10, bold=True)
    rbox(ax, 2, 20, 22, 20, "Computação\n(treinamento)", fc="#D1FAE5", ec=GRN, fs=10, bold=True)
    rbox(ax, 29, 36, 32, 34, "Modelo de linguagem\nP(próximo token | contexto)\n\n\"grande\" quando os três\ningredientes escalam", fc=IND, ec=IND, tc="white", fs=8.8, bold=True)
    for y in (82, 56, 30): arrow(ax, (24.5, y), (28.5, 53), c=GRY)
    rbox(ax, 70, 62, 28, 24, "Aplicação\nprompt, validação,\nerros, versões", fc="#FEF3C7", ec=AMB, fs=10, bold=True)
    rbox(ax, 70, 30, 28, 24, "Sistema\nrecuperação, ferramentas,\nsupervisão, monitoramento", fc="#FCE7F3", ec=PNK, fs=10, bold=True)
    arrow(ax, (61.5, 58), (69.5, 72), c=INK); arrow(ax, (84, 61.5), (84, 54.5), c=INK)
    rbox(ax, 8, 2, 90, 14, "Limites: respostas inventadas · vieses · conhecimento datado · custo · segurança", fc="#FEE2E2", ec=RED, fs=8.8, bold=True)
    ax.text(47, 96, "Do texto às aplicações", fontsize=12, fontweight="bold", ha="center", color=INK)
    save(fig, "f01_mapa")


def aninhados():
    fig, ax = canvas(7.6, 5.4, (0, 100), (0, 72))
    ax.add_patch(Ellipse((38, 36), 70, 66, fc="#E0E7FF", ec=IND, lw=2)); ax.text(38, 63, "Inteligência Artificial", ha="center", fontsize=10.5, fontweight="bold", color=IND)
    ax.add_patch(Ellipse((38, 30), 52, 44, fc="#C7D2FE", ec=IND, lw=1.5)); ax.text(38, 48, "Aprendizado de máquina", ha="center", fontsize=10, fontweight="bold", color=IND)
    ax.add_patch(Ellipse((38, 24), 34, 24, fc="#A5B4FC", ec=IND, lw=1.5)); ax.text(38, 31, "Aprendizado profundo", ha="center", fontsize=9.5, fontweight="bold", color=INDD if False else "#312E81")
    ax.add_patch(Ellipse((75, 24), 40, 30, fc=AMB, ec=AMB, alpha=0.25, lw=0)); ax.add_patch(Ellipse((75, 24), 40, 30, fc="none", ec=AMB, lw=2))
    ax.text(86, 36, "PLN", ha="center", fontsize=11, fontweight="bold", color="#92400E")
    ax.add_patch(Ellipse((57, 22), 16, 14, fc=RED, ec="#B91C1C", lw=1.5, alpha=0.9)); ax.text(57, 22, "LLMs", ha="center", va="center", fontsize=10, fontweight="bold", color="white")
    ax.text(38, 17, "redes\nneurais", ha="center", fontsize=8.5, color="#312E81")
    ax.text(86, 12, "regras,\nestatística…", ha="center", fontsize=8.5, color="#92400E")
    save(fig, "f01_aninhados")


def historia():
    fig, ax = canvas(10, 4.6, (0, 100), (-5, 44))
    ax.plot([3, 97], [22, 22], c=INK, lw=2)
    ev = [(1948, "Shannon:\nteoria da\ninformação"), (2003, "Modelo neural\nprobabilístico"), (2013, "word2vec"), (2015, "Atenção em\ntradução"), (2017, "Transformer"), (2018, "ELMo,\nBERT"), (2020, "GPT-3,\nleis de escala"), (2022, "Ajuste por\npreferências")]
    pos = np.array([4, 24, 38, 50, 61, 72, 83, 94])
    cols = [GRY, IND, CYA, GRN, AMB, RED, VIO, PNK]
    for k, ((ano, txt), x, c) in enumerate(zip(ev, pos, cols)):
        up = k % 2 == 0
        ax.scatter([x], [22], s=110, c=c, zorder=3, edgecolors="white")
        ax.plot([x, x], [22, 31 if up else 13], c=c, lw=1.5)
        ax.text(x, 33 if up else 11, f"{ano}", ha="center", va="bottom" if up else "top", fontsize=11, fontweight="bold", color=c)
        ax.text(x, 37.5 if up else 6.5, txt, ha="center", va="bottom" if up else "top", fontsize=8.3, color=INK, linespacing=1.15)
    ax.text(50, -3.5, "Escala de tempo qualitativa: a distância entre marcos não é proporcional aos anos.", ha="center", fontsize=8, color=GRY, style="italic")
    save(fig, "f01_historia")


def camadas():
    fig, ax = canvas(9, 5.6)
    ys = [74, 52, 30, 8]; nomes = ["Interface de conversação", "Aplicação", "Sistema completo", "Modelo"]
    desc = ["janela de chat, voz, API", "monta o prompt, valida e trata erros", "recuperação, ferramentas, registros, supervisão", "parâmetros treinados: tokens → probabilidades"]
    err = ["instruções ocultas; contexto que o usuário não vê", "prompt mal formulado; saída sem validação", "documento errado recuperado; ação indevida", "conhecimento ausente; viés; resposta inventada"]
    cols = [CYA, AMB, PNK, IND]
    for y, n, d, e, c in zip(ys, nomes, desc, err, cols):
        rbox(ax, 4, y, 44, 16, "", fc=c, ec=c, alpha=0.18)
        ax.add_patch(FancyBboxPatch((4, y), 44, 16, boxstyle="round,pad=0,rounding_size=1.5", fc="none", ec=c, lw=2))
        ax.text(26, y + 10.5, n, ha="center", fontsize=10.5, fontweight="bold", color=INK); ax.text(26, y + 4.5, d, ha="center", fontsize=8.6, color=INK)
        rbox(ax, 54, y + 1.5, 43, 13, "", fc="#FEE2E2", ec="#FCA5A5", lw=1)
        ax.text(75.5, y + 8, "Pode falhar:\n" + e, ha="center", va="center", fontsize=8.2, color="#7F1D1D", linespacing=1.2)
    arrow(ax, (1.8, 82), (1.8, 14), c=INK, lw=2); ax.text(0.4, 48, "pergunta →", rotation=90, fontsize=8, va="center", color=INK)
    save(fig, "f01_camadas")


def expectativas():
    fig, ax = canvas(9, 4.4)
    rbox(ax, 2, 12, 44, 76, "", fc="#ECFDF5", ec=GRN, lw=2)
    ax.text(24, 84, "Em geral útil, com verificação", ha="center", fontsize=11, fontweight="bold", color="#065F46")
    for i, t in enumerate(["reescrever e resumir textos curtos", "classificar e extrair, validando o formato", "responder com base em documentos", "gerar rascunhos para revisão humana", "apoiar programação, com testes"]):
        ax.text(5, 72 - i * 11.5, "✓ " + t, fontsize=8.4, color=INK)
    rbox(ax, 54, 12, 44, 76, "", fc="#FEF2F2", ec=RED, lw=2)
    ax.text(76, 84, "Exige cautela redobrada", ha="center", fontsize=11, fontweight="bold", color="#991B1B")
    for i, t in enumerate(["fatos recentes sem fonte atualizada", "números, citações e referências", "decisões de alto risco sem supervisão", "contas e lógica longas sem checagem", "dados pessoais e sigilosos"]):
        ax.text(57, 72 - i * 11.5, "! " + t, fontsize=8.4, color=INK)
    ax.annotate("", xy=(54, 50), xytext=(46, 50), arrowprops=dict(arrowstyle="<->", color=INK, lw=2))
    ax.text(50, 60, "medir", ha="center", fontsize=8, fontweight="bold", color=INK)
    save(fig, "f01_expectativas")


def ppl():
    frases = corpus.frases_lm(3000, seed=0); tr, te = frases[:2400], frases[2400:]
    vals = [modelos.NgramaLM(n=n, k=0.05).ajustar(tr).perplexidade(te) for n in (1, 2, 3)]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    b = ax.bar(["unigrama\n(n=1)", "bigrama\n(n=2)", "trigrama\n(n=3)"], vals, color=[GRY, CYA, IND], width=0.55)
    for r, v in zip(b, vals): ax.text(r.get_x() + r.get_width() / 2, v + 0.6, f"{v:.2f}", ha="center", fontweight="bold")
    ax.set_ylabel("Perplexidade no teste (menor é melhor)"); ax.set_ylim(0, max(vals) * 1.15)
    ax.set_title("Mais contexto, menos surpresa (corpus sintético)")
    save(fig, "f01_ppl")


if __name__ == "__main__":
    mapa(); aninhados(); historia(); camadas(); expectativas(); ppl()
