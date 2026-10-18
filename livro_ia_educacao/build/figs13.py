from fe import *

def f13_mapa():
    mapa_conceitos("f13_mapa", "IA nos anos\nfinais", [
        ("Estudantes", ["autonomia e identidade", "vida digital"]),
        ("Condições de uso", ["idade e normas", "mediação e registro"]),
        ("Apoios ao professor", ["sequências e exemplos", "rubricas e feedback"]),
        ("Atividades", ["auditoria de respostas", "dica ou resposta?"]),
        ("Avaliação e autoria", ["processo e conversa", "sem detectores"]),
        ("Riscos", ["deepfakes, cyberbullying", "dependência"]),
    ], w=8.6, h=6.0, cor=BLUE)

def f13_sintese():
    fig, ax = canvas(8.4, 4.0, (0, 100), (0, 47))
    cards = [("O que eu\nquero aprender?", TERRA), ("A IA ajuda ou\nfaz por mim?", TEAL), ("Como vou verificar\ne declarar?", PLUM)]
    for i, (t, c) in enumerate(cards):
        x = 4 + i * 32
        rbox(ax, x, 22, 28, 20, t, fc=c, tc="white", bold=True, fs=8.6, r=3, z=3)
        if i < 2: arrow(ax, (x + 28.3, 32), (x + 31.7, 32), INK, lw=1.8)
    grupo(ax, 22, 3, 8, 0.8, 9.5, kinds=["jovem", "jovem", "crianca", "jovem", "adulto", "jovem", "jovem", "crianca"], seed=3)
    save(fig, "f13_sintese")

if __name__ == "__main__":
    f13_mapa(); f13_sintese()
