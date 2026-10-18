from fe import *

def f31_mapa():
    mapa_conceitos("f31_mapa", "Diagnóstico,\nformação e\npilotos", [
        ("Por que diagnosticar", ["problema antes da solução", "linha de base"]),
        ("Dimensões e instrumentos", ["questionário curto", "rodas de conversa"]),
        ("Formação docente", ["sustentada e coletiva", "prática guiada"]),
        ("Desenho do piloto", ["questão e escopo", "critérios prévios"]),
        ("Limites das evidências", ["voluntários", "efeito de novidade"]),
        ("Decisão", ["continuar", "modificar ou interromper"]),
    ], w=8.6, h=6.0, cor=GREEN)

def f31_diag():
    comparar_cartoes("f31_diag", [
        ("Necessidades pedagógicas", ["Que dificuldades são prioritárias?", "A IA é necessária aqui?"], TERRA),
        ("Usos atuais", ["O que já se faz, e como?", "Há uso informal?"], TEAL),
        ("Conhecimento e atitudes", ["O que sabem e esperam?", "O que temem?"], PLUM),
        ("Infraestrutura e acesso", ["Conexão e dispositivos?", "Quem fica de fora?"], BLUE),
        ("Dados e normas", ["Que dados circulam?", "Quem responde por quê?"], ROSE),
        ("Apoio e tempo", ["Há planejamento coletivo?", "Há suporte técnico?"], GOLD),
    ], w=8.6, h=4.8, wrap_at=24, colunas=3)

def f31_formacao():
    camadas("f31_formacao", [
        ("Comunidade de prática e autonomia", "troca, revisão e decisão do grupo"),
        ("Prática guiada com acompanhamento", "atividades da própria disciplina, observação entre pares"),
        ("Sensibilização e escuta", "conhecer usos, dúvidas e critérios"),
    ][::-1], w=8.0, h=4.6, base_larga=True)

def f31_piloto():
    ciclo("f31_piloto", [("Diagnosticar", "necessidades e linha de base"), ("Desenhar", "questão e critérios prévios"),
          ("Formar e executar", "pequena escala"), ("Observar", "registros e retorno"),
          ("Decidir", "continuar, modificar ou interromper")], centro="Piloto", w=7.4, h=5.6, cor=GREEN)

def f31_sintese():
    fig, ax = canvas(8.6, 4.2, (0, 100), (0, 49))
    xs = [2, 25, 48]
    ts = ["Diagnóstico", "Formação", "Piloto"]
    cs = [TERRA, TEAL, BLUE]
    for x, t, c in zip(xs, ts, cs):
        rbox(ax, x, 28, 20, 12, t, fc=c, tc="white", bold=True, fs=8.8, r=3, z=3)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + 20.3, 34), (b - 0.3, 34))
    rbox(ax, 71, 28, 27, 12, "Decisão explícita", fc=GREEN, tc="white", bold=True, fs=8.8, r=3, z=3)
    arrow(ax, (68.3, 34), (70.7, 34))
    for x, t, c in [(60, "Continuar", GREEN), (72, "Modificar", AMBER), (84, "Interromper", ROSE)]:
        rbox(ax, x + 1, 8, 11.5, 8, t, fc=c, tc="white", bold=True, fs=7.8, r=2.5, z=3)
        arrow(ax, (84.5, 27.7), (x + 6.7, 16.4), c=c, lw=1.4)
    rbox(ax, 2, 6, 48, 12, "Em todas as etapas: escuta, ética de dados e equidade", fc=SAND, tc="black", bold=True, fs=8.2, r=3, z=3, wrap_at=40)
    save(fig, "f31_sintese")

if __name__ == "__main__":
    f31_mapa(); f31_diag(); f31_formacao(); f31_piloto(); f31_sintese()
