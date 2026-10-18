from fe import *

def f24_mapa():
    mapa_conceitos("f24_mapa", "Anatomia\ndo pedido", [
        ("Papel", ["quem responde", "para quem"]),
        ("Contexto", ["turma e objetivo", "material de apoio"]),
        ("Tarefa", ["verbo claro", "uma coisa por vez"]),
        ("Formato", ["estrutura", "extensão e linguagem"]),
        ("Critérios", ["o que é bom", "o que evitar"]),
        ("Limites", ["verificar sempre", "dados protegidos"]),
    ], w=8.6, h=6.0, cor=BLUE)

def f24_anatomia():
    fig, ax = canvas(8.6, 4.8, (0, 100), (0, 56))
    itens = [("Papel", "Que perspectiva e que público?", TERRA),
             ("Contexto", "Turma, objetivo, material, restrições", TEAL),
             ("Tarefa", "O que fazer, em um verbo claro", AMBER),
             ("Formato", "Como entregar: estrutura, tamanho, tom", PLUM),
             ("Critérios", "Como saber se ficou bom", BLUE)]
    for i, (t, d, c) in enumerate(itens):
        y = 46 - i * 9.4
        rbox(ax, 2, y, 17, 7.6, t, fc=c, tc="white", bold=True, fs=8.8, r=2, z=3)
        rbox(ax, 20, y, 40, 7.6, d, fc=WHITE, ec=c, fs=8, r=2, lw=1.5, z=3, wrap_at=48)
        arrow(ax, (60.4, y + 3.8), (66.5, 26), c, lw=1.3, style="-", z=1)
    rbox(ax, 66.5, 17, 31, 18, "Resposta\nrascunho\n(a ser verificada)", fc=SAND, ec=GOLD, fs=9, bold=True, r=3, lw=2, z=3)
    rbox(ax, 66.5, 4, 31, 9, "Julgamento\ndocente", fc=GREEN, tc="white", bold=True, fs=8.8, r=3, z=3)
    arrow(ax, (82, 16.6), (82, 13.4), INK, lw=1.5)
    save(fig, "f24_anatomia")

def f24_vago_preciso():
    comparar_cartoes("f24_vago_preciso", [
        ("Pedido vago", ["\"Faça uma aula sobre frações.\"", "Resultado: texto genérico, sem série, sem objetivo, sem formato.", "Quem decide tudo é o sistema."], ROSE),
        ("Pedido com lacunas", ["Informa a série, mas não o objetivo nem o formato.", "Resultado: bom tema, atividade pouco alinhada.", "Exige várias rodadas de correção."], AMBER),
        ("Pedido bem formulado", ["Papel, contexto, tarefa, formato e critérios.", "Resultado: rascunho útil para ajustar.", "O professor continua decidindo e verificando."], GREEN),
    ], w=8.6, h=4.0, wrap_at=30)

def f24_sintese():
    fig, ax = canvas(8.6, 3.8, (0, 100), (0, 44))
    xs = [2, 26, 50, 74]
    ts = ["Objetivo\ndocente", "Pedido\nestruturado", "Rascunho\nda IA", "Verificação\ne decisão"]
    cs = [TEAL, BLUE, AMBER, TERRA]
    for x, t, c in zip(xs, ts, cs):
        rbox(ax, x, 20, 22, 14, t, fc=c, tc="white", bold=True, fs=8.8, r=3, z=3)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + 22.2, 27), (b - 0.2, 27))
    arrow(ax, (85, 19.5), (13, 19.5), GRY, lw=1.3, rad=0.35, ls="--")
    label(ax, 50, 8.5, "Se não serve, volte e ajuste o pedido ou descarte o rascunho", fs=8.2, it=True)
    save(fig, "f24_sintese")

if __name__ == "__main__":
    f24_mapa(); f24_anatomia(); f24_vago_preciso(); f24_sintese()
