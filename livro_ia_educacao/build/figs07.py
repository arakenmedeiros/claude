from fe import *

def f07_mapa():
    mapa_conceitos("f07_mapa", "Como se aprende", [
        ("Conhecimentos prévios", ["ancoragem", "concepções alternativas"]),
        ("Desenvolvimento e mediação", ["ZDP e andaimes", "idade importa"]),
        ("Carga cognitiva", ["memória de trabalho", "carga relevante"]),
        ("Esforço produtivo", ["dificuldades desejáveis", "efeito do teste"]),
        ("Aprendizagem ativa", ["com orientação", "colaboração"]),
        ("Diferenças individuais", ["sem rótulos", "estilos: sem base"]),
    ], w=8.6, h=6.0, cor=GOLD)

def f07_zdp():
    fig, ax = canvas(8.0, 4.6, (0, 100), (0, 58))
    cx, cy = 34, 29
    for r, c, t, ty in [(27, "#F6E7D3", "ainda não alcança", cy + 22.5), (18.5, "#F8D9B8", "faz com ajuda\n(ZDP)", cy + 12.8), (9.5, TERRA, "faz\nsozinho", cy)]:
        ax.add_patch(Circle((cx, cy), r, fc=c, ec="white", lw=2, zorder=2))
        ax.text(cx, ty, t, ha="center", va="center", fontsize=8.2, fontweight="bold", color="white" if r < 10 else INK, zorder=5, linespacing=1.05)
    rbox(ax, 66, 24, 32, 30, "", fc=SAND, ec=GOLD, lw=1.6, r=2, z=2)
    ax.text(82, 39, wrap("A ajuda adequada se ajusta ao estudante e é retirada gradualmente. Se a ajuda faz a tarefa por ele, a zona de aprendizagem desaparece.", 30), ha="center", va="center", fontsize=7.8, zorder=5, linespacing=1.25)
    arrow(ax, (63, 30), (46, 38), GOLD, lw=1.6, rad=0.2)
    save(fig, "f07_zdp")

def f07_esforco():
    fig, ax = canvas(8.6, 3.2, (0, 100), (0, 37))
    zonas = [(3, 30, "Fácil demais ou\nfeito por outro", "pouco esforço,\npouca aprendizagem", AMBER), (34, 33, "Esforço produtivo", "desafio possível\ncom apoio adequado", GREEN), (68, 29, "Impossível\nou confuso", "esforço improdutivo,\nfrustração", ROSE)]
    for x, w, t, d, c in zonas:
        rbox(ax, x, 11, w, 20, "", fc=c, r=2, z=2, alpha=0.92)
        ax.text(x + w / 2, 25.2, t, ha="center", va="center", fontsize=8.6, fontweight="bold", color="white", zorder=5, linespacing=1.1)
        ax.text(x + w / 2, 16.5, d, ha="center", va="center", fontsize=7.6, color="white", zorder=5, linespacing=1.1)
    arrow(ax, (6, 6), (94, 6), INK, lw=1.4, style="<|-|>")
    label(ax, 50, 2.2, "a IA pode deslocar o estudante para qualquer ponto desta escala", 7.8, GRY, it=True)
    save(fig, "f07_esforco")

def f07_sintese():
    fig, ax = canvas(8.6, 3.8, (0, 100), (0, 44))
    painel = [(3, "Tende a ajudar", GREEN, "#EAF3E3", ["Dicas graduais e perguntas", "Exemplos e reformulações", "Feedback após a tentativa", "Variação de contextos"]),
              (52, "Tende a atrapalhar", ROSE, "#FADDE5", ["Resposta pronta antes da tentativa", "Substituição do raciocínio", "Explicações não verificadas", "Rótulos sobre estudantes"])]
    for x, t, c, f, ls in painel:
        rbox(ax, x, 3, 45, 38, "", fc=f, ec=c, lw=2, r=3, z=2)
        rbox(ax, x, 33, 45, 8, t, fc=c, tc="white", bold=True, fs=9.5, r=3, z=3)
        for i, l in enumerate(ls):
            ax.text(x + 4, 28 - i * 6.2, "• " + l, ha="left", va="center", fontsize=8, zorder=5)
    label(ax, 50, 0.4, "Ilustração conceitual com base em princípios, não em dados.", 7, GRY, it=True)
    save(fig, "f07_sintese")

if __name__ == "__main__":
    f07_mapa(); f07_zdp(); f07_esforco(); f07_sintese()
