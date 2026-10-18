from fe import *

def f17_mapa():
    mapa_conceitos("f17_mapa", "IA na\npós-graduação", [
        ("Normas e recomendações", ["ICMJE", "políticas do CNPq"]),
        ("Fases da pesquisa", ["revisão", "análise", "escrita"]),
        ("Verificação em cadeia", ["fonte primária", "rastreabilidade"]),
        ("Confidencialidade", ["dados de participantes", "pareceres"]),
        ("Autoria e declaração", ["responsabilidade", "registro do uso"]),
        ("Orientação", ["acordos claros", "avaliação do processo"]),
    ], w=8.6, h=6.0, cor=PLUM)

def f17_sintese():
    fig, ax = canvas(8.6, 4.0, (0, 100), (0, 46))
    xs = [2, 26, 50, 74]
    ts = ["Fonte lida\n(primária)", "Afirmação\nrastreável", "Texto\nautoral", "Declaração\nde uso"]
    cs = [TEAL, BLUE, TERRA, PLUM]
    for x, t, c in zip(xs, ts, cs):
        rbox(ax, x, 22, 22, 14, t, fc=c, tc="white", bold=True, fs=8.6, r=3, z=3)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + 22.2, 29), (b - 0.2, 29))
    rbox(ax, 2, 3, 94, 11, "Escudos: confidencialidade  •  ética em pesquisa  •  normas da revista e da instituição", fc=SAND, tc="black", bold=True, fs=8.6, r=3, z=3)
    save(fig, "f17_sintese")

if __name__ == "__main__":
    f17_mapa(); f17_sintese()
