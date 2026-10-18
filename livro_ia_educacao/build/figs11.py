from fe import *

def f11_mapa():
    mapa_conceitos("f11_mapa", "IA e educação\ninfantil", [
        ("Direitos e campos", ["conviver, brincar", "explorar, expressar"]),
        ("Uso docente", ["ideias e rascunhos", "revisão, sem dados"]),
        ("Participação das crianças", ["sem assistentes", "demonstração mediada"]),
        ("Atividades desplugadas", ["robô de comando", "pessoas e máquinas"]),
        ("Avaliação", ["observação e registro", "documentação"]),
        ("Riscos e proteção", ["privacidade", "antropomorfismo"]),
    ], w=8.6, h=6.0, cor=GOLD)

def f11_usos():
    comparar_cartoes("f11_usos", [
        ("Uso pelo professor (preparação)", ["Ideias de histórias, brincadeiras e músicas", "Rascunhos de comunicados", "Organização de notas sem dados identificáveis", "Sempre com revisão"], TEAL),
        ("Uso direto pelas crianças", ["Em geral, não recomendado com assistentes generativos", "Prioridade: experiência concreta e brincadeira", "Conversar sobre máquinas com atividades desplugadas"], ROSE)],
        w=8.6, h=3.4, wrap_at=40)

def f11_desplugado():
    fig, ax = canvas(8.6, 3.4, (0, 100), (0, 40))
    cenas = [("Robô de comando", TERRA), ("Pessoas ou máquinas?", TEAL), ("Ordenar cenas", PLUM), ("Roda de conversa", BLUE)]
    for i, (t, c) in enumerate(cenas):
        x = 2 + i * 24.5
        rbox(ax, x, 3, 23, 34, "", fc=WHITE, ec=c, lw=2.2, r=3, z=2)
        rbox(ax, x, 30, 23, 7, t, fc=c, tc="white", bold=True, fs=7.6, r=3, z=3)
        if i == 0:
            pessoa(ax, x + 7, 8, 0.8, 1, 1, TERRA, "crianca"); pessoa(ax, x + 16, 8, 0.8, 3, 2, AMBER, "crianca")
            arrow(ax, (x + 9.5, 22), (x + 14.5, 22), INK, lw=1.4)
        elif i == 1:
            robo(ax, x + 6, 9, 0.8, BLUE); pessoa(ax, x + 16, 9, 0.9, 2, 3, ROSE, "adulto")
        elif i == 2:
            for k in range(3):
                rbox(ax, x + 2.5 + k * 6.8, 11, 5.6, 9, str(k + 1), fc=SAND, ec=c, fs=8, bold=True, r=1, z=4)
        else:
            for k in range(5):
                pessoa(ax, x + 3.5 + k * 4, 9 + (k % 2) * 3, 0.55, k, k, CL[k % 8], "crianca")
    save(fig, "f11_desplugado")

def f11_sintese():
    fig, ax = canvas(8.4, 4.4, (0, 100), (0, 52))
    ax.add_patch(Circle((50, 20), 15, fc=SAND, ec=GOLD, lw=2, zorder=1))
    kinds = ["crianca", "crianca", "jovem", "crianca", "crianca", "adulto"]
    import math
    for i in range(6):
        a = math.pi / 2 - i * 2 * math.pi / 6
        pessoa(ax, 50 + 12 * math.cos(a), 20 + 8.5 * math.sin(a) - 4, 0.8, i, i + 1, CL[i % 8], kinds[i])
    rbox(ax, 4, 38, 40, 11, "IA nos bastidores: ideias e rascunhos, sem dados pessoais, com revisão", fc=WHITE, ec=TEAL, lw=1.8, r=2.5, fs=7.6, wrap_at=34, z=3)
    ax.plot([2, 98], [36, 36], c=MID, lw=1.5, ls="--")
    for x, t in [(14, "brincar"), (50, "conviver"), (86, "explorar")]:
        pill(ax, x, 3.5, t, fc=[TERRA, PLUM, TEAL][[14, 50, 86].index(x)], w=20, h=5, fs=8.5)
    save(fig, "f11_sintese")

if __name__ == "__main__":
    f11_mapa(); f11_usos(); f11_desplugado(); f11_sintese()
