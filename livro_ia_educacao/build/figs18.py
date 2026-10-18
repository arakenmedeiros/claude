from fe import *

def f18_mapa():
    mapa_conceitos("f18_mapa", "Planejar e criar\nmateriais com IA", [
        ("Ponto de partida", ["objetivos", "evidências", "turma"]),
        ("O que delegar", ["delegar", "parceria", "reservar"]),
        ("Como pedir", ["contexto", "restrições", "formato"]),
        ("Materiais", ["adaptar", "exercícios", "acessibilidade"]),
        ("Verificação", ["resolver itens", "conferir fatos"]),
        ("Riscos e cuidados", ["erros fluentes", "dados e direitos"]),
    ], w=8.6, h=6.0, cor=PLUM)

def f18_fluxo():
    fluxo("f18_fluxo", [
        ("Objetivos", "currículo e turma: decisão docente"),
        ("Evidências", "o que mostrará que aprenderam"),
        ("Sequência", "atividades e ordem; apoio possível da IA"),
        ("Materiais", "textos, exercícios, rubricas; apoio com revisão"),
        ("Teste e ajuste", "aplicar, observar, replanejar"),
    ], w=8.6, cor="multi", wrap_at=15, por_linha=5, h=2.6)

def f18_semaforo():
    semaforo("f18_semaforo", [
        ("v", "Delegar com revisão rápida", "reformatar texto seu, variar enunciados, sugerir títulos"),
        ("a", "Fazer em parceria, com revisão atenta", "sequências, equívocos comuns, exercícios com conferência total"),
        ("r", "Reservar ao professor", "objetivos, critérios, adaptações individuais, avaliação do que a turma compreendeu"),
    ], w=8.6, wrap_at=70)

def f18_ciclo():
    ciclo("f18_ciclo", [("Planejar", "objetivos e evidências"), ("Aplicar", "com a turma real"),
                        ("Observar", "o que a turma mostra"), ("Ajustar", "registrar e replanejar")],
          centro="Decisão\ndocente", w=7.2, h=5.0, cor=TEAL)

def f18_sintese():
    fig, ax = canvas(8.6, 4.0, (0, 100), (0, 46))
    xs = [2, 26, 50, 74]
    ts = ["Objetivos e\nevidências", "Rascunho\n(apoio da IA)", "Verificação e\nadaptação", "Uso em aula\ne ajuste"]
    cs = [PLUM, BLUE, TEAL, TERRA]
    for x, t, c in zip(xs, ts, cs):
        rbox(ax, x, 22, 22, 14, t, fc=c, tc="white", bold=True, fs=8.6, r=3, z=3)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + 22.2, 29), (b - 0.2, 29))
    label(ax, 13, 39.5, "decisão docente", 8, PLUM, True); label(ax, 37, 39.5, "apoio, não decisão", 8, BLUE, True)
    label(ax, 61, 39.5, "decisão docente", 8, TEAL, True); label(ax, 85, 39.5, "decisão docente", 8, TERRA, True)
    rbox(ax, 2, 3, 94, 11, "Cuidados: dados de estudantes protegidos  •  acessibilidade  •  direitos autorais", fc=SAND, tc="black", bold=True, fs=8.6, r=3, z=3)
    save(fig, "f18_sintese")

if __name__ == "__main__":
    f18_mapa(); f18_fluxo(); f18_semaforo(); f18_ciclo(); f18_sintese()
