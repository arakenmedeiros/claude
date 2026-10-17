from fe import *

def f06_mapa():
    mapa_conceitos("f06_mapa", "Objetivos e\nalinhamento", [
        ("O que é um objetivo", ["observável, específico", "relevante e viável"]),
        ("Bloom revisada", ["tipo de conhecimento", "processo cognitivo"]),
        ("Alinhamento construtivo", ["objetivo, atividade", "avaliação coerentes"]),
        ("De trás para frente", ["resultados e evidências", "depois o plano"]),
        ("Objetivos e IA", ["conteúdo, letramento", "trabalho do professor"]),
        ("Verificação", ["três perguntas", "revisão e registro"]),
    ], w=8.6, h=6.0, cor=TEAL)

def f06_bloom():
    fig, ax = canvas(8.6, 4.4, (0, 100), (0, 51))
    niv = [("Lembrar", "listar, nomear, recordar"), ("Compreender", "explicar, resumir, exemplificar"), ("Aplicar", "resolver, executar, usar"), ("Analisar", "comparar, relacionar, distinguir"), ("Avaliar", "julgar, criticar, justificar"), ("Criar", "projetar, produzir, propor")]
    cores = [AMBER, "#EE9B4C", CORAL, TERRA, "#A94463", PLUM]
    for i, ((t, d), c) in enumerate(zip(niv, cores)):
        y = 3 + i * 7.2
        rbox(ax, 3, y, 58, 6.4, "", fc=c, r=1.2, z=2)
        ax.text(5, y + 3.2, t, ha="left", va="center", fontsize=8.6, fontweight="bold", color="white", zorder=5)
        ax.text(24, y + 3.2, d, ha="left", va="center", fontsize=7.4, color="white", zorder=5)
    rbox(ax, 66, 3, 32, 40, "", fc=SAND, ec=GOLD, lw=1.6, r=2, z=2)
    ax.text(82, 25, wrap("Em todos os degraus, a IA pode apoiar o raciocínio ou substituí-lo. O que decide é o desenho da atividade: o que o estudante precisa fazer sozinho para aprender.", 30), ha="center", va="center", fontsize=8, zorder=5, linespacing=1.25)
    save(fig, "f06_bloom")

def f06_triangulo():
    fig, ax = canvas(8.2, 4.6, (0, 100), (0, 56))
    P = {"o": (50, 46), "a": (14, 9), "v": (86, 9)}
    ax.add_patch(Polygon([P["o"], P["a"], P["v"]], fc=SAND, ec=GOLD, lw=2, zorder=1, alpha=0.7))
    for k, t, d, c in [("o", "Objetivos", "o que o estudante\nserá capaz de fazer", TERRA), ("a", "Atividades", "como ele pratica\no processo", TEAL), ("v", "Avaliação", "que evidência\nmostra a aprendizagem", PLUM)]:
        x, y = P[k]
        rbox(ax, x - 13, y - 5, 26, 10, "", fc=c, r=2.5, z=3)
        ax.text(x, y + 1.8, t, ha="center", va="center", fontsize=9, fontweight="bold", color="white", zorder=5)
        ax.text(x, y - 1.9, d, ha="center", va="center", fontsize=6.5, color="white", zorder=5, linespacing=1.1)
    ax.text(50, 22, "coerência", ha="center", va="center", fontsize=11, fontweight="bold", color=GOLD, style="italic", zorder=5)
    save(fig, "f06_triangulo")

def f06_reverso():
    fig, ax = canvas(8.6, 3.0, (0, 100), (0, 35))
    et = [("3. Plano de aprendizagem", "experiências e ensino", BLUE), ("2. Evidências", "desempenhos e produtos", TEAL), ("1. Resultados desejados", "o que compreender e fazer", TERRA)]
    for i, (t, d, c) in enumerate(et):
        x = 3 + i * 33
        rbox(ax, x, 6, 28, 18, "", fc=WHITE, ec=c, lw=2, r=3, z=2)
        rbox(ax, x, 17.5, 28, 6.5, t, fc=c, tc="white", bold=True, fs=8, r=3, z=3)
        ax.text(x + 14, 11, d, ha="center", va="center", fontsize=8, zorder=5)
        if i < 2: arrow(ax, (x + 28.3, 15), (x + 32.7, 15), INK, lw=1.8, style="<|-")
    arrow(ax, (87, 30), (13, 30), TERRA, lw=2.2)
    ax.text(50, 33.2, "ordem de pensamento do professor", ha="center", fontsize=8, color=TERRA, fontweight="bold")
    ax.text(50, 1.5, "a execução em aula segue a ordem inversa: do plano às evidências", ha="center", fontsize=7.4, color=GRY, style="italic")
    save(fig, "f06_reverso")

def f06_fluxo():
    fluxo("f06_fluxo", [("Objetivo", "defino o que e a evidência"), ("Contexto", "sem dados pessoais"), ("Rascunho", "peço à IA"), ("Alinhamento", "aplico as 3 perguntas"), ("Revisão", "fatos, nível, acesso"), ("Aula e balanço", "o que aprenderam?")],
          w=8.6, cor="multi", wrap_at=15, h=3.8)

def f06_sintese():
    fig, ax = canvas(8.2, 3.6, (0, 100), (0, 42))
    rbox(ax, 36, 15, 28, 12, "Objetivo de\naprendizagem", fc=TERRA, tc="white", bold=True, fs=9.5, r=3, z=3)
    for (x, y, t, c) in [(3, 28, "Atividades", TEAL), (3, 4, "Avaliação", PLUM), (72, 15, "Uso (ou não)\nda IA", BLUE)]:
        rbox(ax, x, y, 24, 10, t, fc=c, tc="white", bold=True, fs=8.4, r=2.5, z=3)
    arrow(ax, (36, 24), (27.3, 31), INK, lw=1.8); arrow(ax, (36, 18), (27.3, 9.5), INK, lw=1.8); arrow(ax, (64.3, 21), (71.7, 21), INK, lw=1.8)
    label(ax, 50, 5.5, "se a IA não ajuda o objetivo, pode ser dispensada", 7.8, GRY, it=True)
    save(fig, "f06_sintese")

if __name__ == "__main__":
    f06_mapa(); f06_bloom(); f06_triangulo(); f06_reverso(); f06_fluxo(); f06_sintese()
