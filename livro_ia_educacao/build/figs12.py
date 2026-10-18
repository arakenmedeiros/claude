from fe import *

def f12_mapa():
    mapa_conceitos("f12_mapa", "IA nos anos\niniciais", [
        ("Desenvolvimento", ["alfabetização", "mediação do adulto"]),
        ("Apoios ao professor", ["textos em níveis", "problemas e roteiros"]),
        ("Participação das crianças", ["sem assistentes", "demonstrações mediadas"]),
        ("Atividades", ["próxima palavra", "detetive de fontes"]),
        ("Avaliação", ["observação, portfólio", "feedback descritivo"]),
        ("Riscos", ["cópia e dependência", "privacidade"]),
    ], w=8.6, h=6.0, cor=TERRA)

def f12_sintese():
    fig, ax = canvas(8.2, 4.2, (0, 100), (0, 50))
    pessoa(ax, 50, 20, 1.5, 2, 1, AMBER, "crianca")
    pessoa(ax, 20, 14, 1.3, 3, 4, TEAL, "adulto", glasses=True)
    robo(ax, 86, 24, 0.7, BLUE)
    arrow(ax, (28, 22), (45, 26), TEAL, lw=2)
    arrow(ax, (80, 28), (30, 30), BLUE, lw=1.6, ls="--", rad=-0.25)
    pill(ax, 50, 8, "produzir", TERRA, w=20, h=5, fs=8.5); pill(ax, 20, 5.5, "mediar", TEAL, w=18, h=5, fs=8.5); pill(ax, 86, 14, "revisar", BLUE, w=18, h=5, fs=8.5)
    label(ax, 50, 46, "A criança é a autora; a IA fica nos bastidores do professor", 9, INK, True)
    save(fig, "f12_sintese")

if __name__ == "__main__":
    f12_mapa(); f12_sintese()
