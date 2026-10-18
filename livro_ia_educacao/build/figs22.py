from fe import *

def f22_mapa():
    mapa_conceitos("f22_mapa", "Avaliar e dar\nfeedback com IA", [
        ("Avaliação diagnóstica", ["antes de ensinar", "itens com distratores"]),
        ("Avaliação formativa", ["ciclo evidência-ação", "bilhetes de saída"]),
        ("Rubricas", ["critérios observáveis", "construção com a turma"]),
        ("Feedback", ["próximo passo", "IA rascunha, professor assina"]),
        ("Integridade e limites", ["sem nota automática", "detectores não provam"]),
        ("Evidências", ["efeito varia muito", "IA: estudos iniciais"]),
    ], w=8.6, h=6.0, cor=PLUM)

def f22_ciclo():
    ciclo("f22_ciclo", [
        ("Objetivo e critério", "o que é qualidade?"),
        ("Evidência", "produções, respostas"),
        ("Interpretação e feedback", "onde estamos?"),
        ("Ação de ajuste", "professor e estudante"),
    ], centro="Avaliação\nformativa", w=7.6, h=5.4, cor=TERRA, wrap_at=14)

def f22_sintese():
    fig, ax = canvas(8.6, 4.2, (0, 100), (0, 49))
    xs = [2, 26, 50, 74]
    ts = ["Critérios\nclaros", "Evidências de\naprendizagem", "Feedback com o\npróximo passo", "Ação do\nestudante"]
    cs = [TEAL, BLUE, TERRA, PLUM]
    for x, t, c in zip(xs, ts, cs):
        rbox(ax, x, 15, 22, 14, t, fc=c, tc="white", bold=True, fs=8.4, r=3, z=3)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + 22.2, 22), (b - 0.2, 22))
    rbox(ax, 2, 38, 94, 8, "Decisão e responsabilidade do professor", fc=SAND, tc="black", bold=True, fs=9, r=3, z=3)
    rbox(ax, 30, 2, 42, 8, "IA: rascunhos para o professor revisar", fc=WHITE, ec=BLUE, tc=BLUE, fs=8.4, bold=True, r=3, z=3)
    arrow(ax, (51, 10.2), (51, 14.8), BLUE, ls="--")
    save(fig, "f22_sintese")

if __name__ == "__main__":
    f22_mapa(); f22_ciclo(); f22_sintese()
