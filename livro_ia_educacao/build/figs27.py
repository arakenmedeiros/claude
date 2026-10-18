from fe import *

def f27_mapa():
    mapa_conceitos("f27_mapa", "Benefícios e\nevidências", [
        ("Tipos de estudo", ["relato a experimento", "revisões"]),
        ("Leitura crítica", ["comparação", "medida e duração"]),
        ("O que sabemos", ["tutoria e feedback", "desenho pedagógico"]),
        ("O que não sabemos", ["longo prazo", "educação básica"]),
        ("Desempenho x aprendizagem", ["momento sem IA", "transferência"]),
        ("Decisão local", ["mini-piloto", "continuar, ajustar, parar"]),
    ], w=8.6, h=6.0, cor=ROSE)

def f27_leitura():
    fluxo("f27_leitura", [
        ("Quem", "Quem foi estudado? Idade, nível, país, quantidade."),
        ("O quê", "Que uso de IA e com que orientação pedagógica?"),
        ("Comparado com", "O que o grupo de comparação fazia?"),
        ("Medida", "O que foi medido? Foi sem IA?"),
        ("Quando", "Por quanto tempo? Houve medida tardia?"),
        ("Quanto", "Tamanho do efeito, custo e consistência entre estudos."),
    ], w=8.6, h=3.6, cor="multi", por_linha=3, wrap_at=20)

def f27_escada():
    camadas("f27_escada", [
        ("Propaganda sem fonte", "números sem estudo localizável"),
        ("Relato de experiência", "inspira, mas não prova efeito"),
        ("Estudo com comparação", "sugere diferença, com ressalvas"),
        ("Experimento com sorteio", "estima efeito causal no contexto"),
        ("Vários estudos e réplicas", "confiança maior, ainda com limites"),
    ], w=8.0, h=5.2, base_larga=True)

def f27_desempenho():
    comparar_cartoes("f27_desempenho", [
        ("Com a IA disponível", ["Tarefa concluída", "Texto bem acabado", "Rapidez", "Mede desempenho assistido"], ROSE),
        ("Depois, sem a IA", ["Consegue explicar com as próprias palavras?", "Resolve um problema novo e parecido?", "Lembra uma semana depois?", "Mede aprendizagem"], TEAL),
    ], w=8.6, h=3.8, wrap_at=30)

def f27_sintese():
    fig, ax = canvas(8.6, 4.0, (0, 100), (0, 46))
    xs = [2, 26, 50, 74]
    ts = ["Afirmação\nde benefício", "Fonte\nlocalizada", "Leitura\ncrítica", "Piloto\nlocal"]
    cs = [TERRA, BLUE, PLUM, TEAL]
    for x, t, c in zip(xs, ts, cs):
        rbox(ax, x, 22, 22, 14, t, fc=c, tc="white", bold=True, fs=8.6, r=3, z=3)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + 22.2, 29), (b - 0.2, 29))
    rbox(ax, 2, 3, 94, 11, "Decisão informada: continuar  •  modificar  •  interromper", fc=SAND, tc="black", bold=True, fs=8.6, r=3, z=3)
    save(fig, "f27_sintese")

if __name__ == "__main__":
    f27_mapa(); f27_leitura(); f27_escada(); f27_desempenho(); f27_sintese()
