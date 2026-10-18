from fe import *

def f21_mapa():
    mapa_conceitos("f21_mapa", "Projetos e problemas\ncom apoio de IA", [
        ("Abordagens", ["problemas (PBL)", "projetos", "interdisciplinaridade"]),
        ("Pergunta motriz", ["autêntica e viável", "produto e critérios"]),
        ("Ciclo do projeto", ["marcos e checagens", "campo e dados locais"]),
        ("Mediação docente", ["andaimes", "perguntas de checagem"]),
        ("Trabalho em grupo", ["papéis rotativos", "avaliação individual"]),
        ("Riscos e evidências", ["projeto genérico", "evidências limitadas"]),
    ], w=8.6, h=6.0, cor=PLUM)

def f21_ciclo():
    ciclo("f21_ciclo", [
        ("Pergunta", "o que queremos resolver?"),
        ("Planejamento", "papéis, marcos"),
        ("Investigação", "campo, fontes, dados"),
        ("Produção", "o produto do grupo"),
        ("Apresentação", "público real"),
        ("Reflexão", "o que aprendemos?"),
    ], centro="Mediação e\nchecagem do\nprofessor", w=8.0, h=5.6, cor=TEAL)

def f21_sintese():
    fig, ax = canvas(8.6, 4.6, (0, 100), (-4, 47))
    for (x, c, t) in [(30, TEAL, "Território e\ncomunidade"), (50, TERRA, "Professores\ne currículo"), (40, GOLD, "")]:
        pass
    ax.add_patch(Circle((27, 27), 16, fc=TEAL, alpha=0.30, ec=TEAL, lw=2))
    ax.add_patch(Circle((47, 27), 16, fc=TERRA, alpha=0.30, ec=TERRA, lw=2))
    ax.add_patch(Circle((37, 14), 16, fc=PLUM, alpha=0.30, ec=PLUM, lw=2))
    label(ax, 19, 33, "Território e\ncomunidade", 8.2, INK, True)
    label(ax, 55, 33, "Professores\ne currículo", 8.2, INK, True)
    label(ax, 37, 6, "Estudantes\ne perguntas", 8.2, INK, True)
    label(ax, 37, 23, "Projeto\nautêntico", 9, INK, True)
    rbox(ax, 66, 18, 32, 18, "IA: apoio ao planejamento, à investigação e à reflexão", fc=SAND, ec=BLUE, fs=8.2, bold=True, wrap_at=24, r=3, z=3)
    arrow(ax, (66, 27), (54, 24), BLUE, ls="--", lw=1.6)
    label(ax, 82, 12, "decisões pedagógicas\ncontinuam humanas", 8, GRY, it=True)
    save(fig, "f21_sintese")

if __name__ == "__main__":
    f21_mapa(); f21_ciclo(); f21_sintese()
