from fe import *

def f05_mapa():
    mapa_conceitos("f05_mapa", "Letramento\nem IA e papel\ndo professor", [
        ("O que é letramento", ["compreender e avaliar", "participar de decisões"]),
        ("Quatro movimentos", ["compreender, utilizar", "avaliar, participar"]),
        ("Letramento docente", ["TPACK e juízo profissional", "escada do autor"]),
        ("Estudantes", ["varia por etapa", "dimensão crítica"]),
        ("A IA pode apoiar", ["rascunhos e variações", "sempre com verificação"]),
        ("Permanece humano", ["vínculo e cuidado", "decisão e justiça"]),
    ], w=8.6, h=6.0, cor=TEAL)

def f05_verbos():
    comparar_cartoes("f05_verbos", [
        ("1. Compreender", ["O que é IA", "Como funciona", "O que não faz"], TERRA),
        ("2. Utilizar", ["Com finalidade clara", "Com cuidado e ética", "Com alternativas"], TEAL),
        ("3. Avaliar", ["Qualidade e correção", "Vieses e riscos", "Efeitos na aprendizagem"], PLUM),
        ("4. Participar", ["Decidir e discutir", "Exigir transparência", "Proteger direitos"], BLUE)],
        w=8.6, h=3.3, wrap_at=22)

def f05_escada():
    fig, ax = canvas(8.6, 4.6, (0, 100), (0, 53))
    niveis = [("1. Consciente", "sabe que existe e quais\nsão as principais questões"), ("2. Compreende", "explica como funciona\ne seus limites"), ("3. Experimenta\ncom segurança", "testa com cuidado\ne verifica"),
              ("4. Planeja e avalia", "integra a atividades\ne mede resultados"), ("5. Lidera e forma", "apoia colegas e\nparticipa de políticas")]
    cores = [AMBER, CORAL, TERRA, PLUM, "#4A2D5C"]
    for i, ((t, d), c) in enumerate(zip(niveis, cores)):
        x = 3 + i * 18.4
        h = 10 + i * 6.2
        rbox(ax, x, 3, 17.6, h, "", fc=c, r=1.5, z=2)
        ax.text(x + 8.8, 3 + h - 3.6, t, ha="center", va="center", fontsize=7.5, fontweight="bold", color="white", zorder=5, linespacing=1.05)
        ax.text(x + 8.8, 3 + h / 2 - 2.5, d, ha="center", va="center", fontsize=6.2, color="white", zorder=5, linespacing=1.1)
    rbox(ax, 3, 41.5, 50, 9, "", fc=SAND, ec=GOLD, lw=1.6, r=2, z=2)
    ax.text(28, 46, wrap("Escolher não usar IA, com fundamentação, pode ocorrer em qualquer degrau.", 46), ha="center", va="center", fontsize=7.6, zorder=5, linespacing=1.2)
    label(ax, 97, 50.5, "Proposta do autor, não validada empiricamente", 7, GRY, it=True, ha="right")
    save(fig, "f05_escada")

def f05_papeis():
    comparar_cartoes("f05_papeis", [
        ("Pode receber apoio de IA, com verificação", ["Rascunhos de planos e de exemplos", "Variações de exercícios", "Descrições de imagens (a validar)", "Organização de informações"], TEAL),
        ("Permanece com o professor", ["Conhecer os estudantes", "Criar vínculo e confiança", "Decidir com ética e justiça", "Mediar conflitos e avaliar", "Aprender com a turma"], TERRA)],
        w=8.6, h=4.2, wrap_at=40)

def f05_sintese():
    fig, ax = canvas(8.2, 4.4, (0, 100), (0, 54))
    ax.add_patch(Polygon([[4, 40], [50, 52], [96, 40]], fc=TERRA, ec="none", zorder=2))
    ax.text(50, 44.2, "Decisão pedagógica fundamentada", ha="center", va="center", color="white", fontsize=10, fontweight="bold", zorder=5)
    ax.add_patch(Rectangle((4, 37), 92, 3, fc="#8E3B57", ec="none", zorder=2))
    cols = [("Letramento em IA", "compreender, utilizar,\navaliar, participar", TEAL), ("Juízo profissional", "conteúdo, pedagogia\ne tecnologia", PLUM), ("Vínculo humano", "relação, cuidado\ne confiança", GOLD)]
    for i, (t, d, c) in enumerate(cols):
        x = 9 + i * 30
        rbox(ax, x, 4, 24, 32, "", fc=c, r=1.5, z=2)
        ax.text(x + 12, 28, t, ha="center", va="center", fontsize=8.6, fontweight="bold", color="white", zorder=5, linespacing=1.1)
        ax.text(x + 12, 17, d, ha="center", va="center", fontsize=7.4, color="white", zorder=5, linespacing=1.2)
    ax.add_patch(Rectangle((4, 1.5), 92, 2.5, fc=MID, ec="none", zorder=2))
    save(fig, "f05_sintese")

if __name__ == "__main__":
    f05_mapa(); f05_verbos(); f05_escada(); f05_papeis(); f05_sintese()
