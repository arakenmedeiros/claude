from fe import *

def f14_mapa():
    mapa_conceitos("f14_mapa", "IA no ensino\nmédio", [
        ("Finalidades", ["formação geral", "projeto de vida"]),
        ("Condições de uso", ["declaração", "proteção de dados"]),
        ("Argumentação", ["advogado do diabo", "verificação"]),
        ("Pesquisa", ["matriz de fontes", "síntese própria"]),
        ("Avaliação e feedback", ["rubricas", "notas de IA: cautela"]),
        ("Integridade e riscos", ["processo e autoria", "ansiedade"]),
    ], w=8.6, h=6.0, cor=PLUM)

def f14_sintese():
    fig, ax = canvas(8.4, 4.2, (0, 100), (0, 50))
    pessoa(ax, 15, 8, 1.4, 2, 1, TEAL, "jovem")
    rbox(ax, 28, 28, 28, 14, "Texto do\nestudante", fc=WHITE, ec=TERRA, lw=2, fs=8.6, r=3, z=3, bold=True)
    balao(ax, 62, 30, 34, 14, "Contra-argumento: quais dados sustentam sua tese?", ec=BLUE, fs=7.4, tail=(-0.8, -1), wrap_at=26)
    robo(ax, 88, 8, 0.7, BLUE)
    rbox(ax, 28, 8, 24, 10, "Verificação\nde fontes", fc=PLUM, tc="white", bold=True, fs=8, r=2.5, z=3)
    rbox(ax, 56, 8, 24, 10, "Declaração\nde uso", fc=GREEN, tc="white", bold=True, fs=8, r=2.5, z=3)
    arrow(ax, (42, 28), (40, 18.4), INK, lw=1.6)
    pill(ax, 15, 44, "autoria do estudante", TERRA, w=30, h=5, fs=8)
    save(fig, "f14_sintese")

if __name__ == "__main__":
    f14_mapa(); f14_sintese()
