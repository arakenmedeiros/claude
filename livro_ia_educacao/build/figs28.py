from fe import *

def f28_mapa():
    mapa_conceitos("f28_mapa", "Equidade, vieses\ne segurança", [
        ("Equidade", ["acesso", "uso e resultados"]),
        ("Vieses", ["dados e medidas", "teste de pares"]),
        ("Segurança", ["erros e bajulação", "conteúdo sintético"]),
        ("Adequação etária", ["direitos", "ECA Digital e LGPD"]),
        ("Inclusão", ["acessibilidade", "apoio humano"]),
        ("Protocolos", ["fluxo de incidentes", "revisão periódica"]),
    ], w=8.6, h=6.0, cor=ROSE)

def f28_equidade():
    comparar_cartoes("f28_equidade", [
        ("Acesso", ["Dispositivos", "Conexão estável", "Língua e formato", "Energia e plano de dados"], TERRA),
        ("Uso", ["Letramento em IA", "Orientação de adultos", "Tempo e espaço de estudo", "Repertório prévio"], TEAL),
        ("Resultados", ["Quem aprende mais?", "Quem fica para trás?", "Acompanhar por grupos", "Sem expor estudantes"], PLUM),
    ], w=8.6, h=3.6, wrap_at=22)

def f28_vieses():
    ciclo("f28_vieses", ["Dados de\ntreinamento", "Rotulagem e\nmedidas", "Projeto do\nsistema", "Uso e\ninterpretação", "Decisões que\ngeram dados"], centro="Onde o\nviés entra", w=7.6, h=5.6, cor=ROSE)

def f28_faixas():
    semaforo("f28_faixas", [
        ("r", "Educação infantil", "Sem uso direto pelas crianças; IA, se houver, só na preparação do professor."),
        ("a", "Anos iniciais", "Uso mediado e coletivo pelo professor; sem contas individuais nem dados pessoais."),
        ("a", "Anos finais e ensino médio", "Uso supervisionado, com letramento, acordos e autonomia progressiva; conferir idade mínima e termos."),
        ("v", "Educação superior e adultos", "Uso declarado e responsável, com verificação, sem dados de terceiros."),
    ], w=8.6, wrap_at=88)

def f28_sintese():
    fig, ax = canvas(8.6, 4.4, (0, 100), (0, 51))
    xs = [2, 26, 50, 74]
    ts = ["Equidade\nde acesso", "Teste de\nvieses", "Proteção\npor idade", "Acessibilidade\ncom apoio humano"]
    cs = [TERRA, BLUE, PLUM, TEAL]
    for x, t, c in zip(xs, ts, cs):
        rbox(ax, x, 4, 22, 22, t, fc=c, tc="white", bold=True, fs=8.6, r=3, z=3)
    rbox(ax, 2, 30, 94, 16, "Uso responsável", fc=SAND, ec=GOLD, tc="black", bold=True, fs=11, r=4, z=3, lw=2)
    save(fig, "f28_sintese")

if __name__ == "__main__":
    f28_mapa(); f28_equidade(); f28_vieses(); f28_faixas(); f28_sintese()
