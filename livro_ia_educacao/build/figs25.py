from fe import *

def f25_mapa():
    mapa_conceitos("f25_mapa", "Refinar e\nrevisar", [
        ("Ciclo de refinamento", ["pedir e avaliar", "ajustar uma coisa"]),
        ("Encadear pedidos", ["etapas curtas", "revisão entre etapas"]),
        ("Dar exemplos", ["bom exemplo", "contraexemplo"]),
        ("Definir critérios", ["lista de verificação", "rubrica"]),
        ("Revisar e corrigir", ["apontar o erro", "conferir a correção"]),
        ("Limites", ["bajulação", "quando parar"]),
    ], w=8.6, h=6.0, cor=PLUM)

def f25_ciclo():
    ciclo("f25_ciclo", [("Pedir", "cinco componentes"), ("Ler com critério", "lista de verificação"),
                        ("Diagnosticar", "o que falhou?"), ("Ajustar", "uma mudança por vez"),
                        ("Registrar", "versão e decisão")], centro="Ciclo de\nrefinamento", w=8.0, h=5.8, cor=BLUE)

def f25_diagnostico():
    comparar_cartoes("f25_diagnostico", [
        ("Faltou contexto", ["Sintoma: texto genérico.", "Ajuste: acrescentar turma, objetivo, tempo, recursos."], TERRA),
        ("Tarefa confusa", ["Sintoma: respondeu outra coisa.", "Ajuste: dividir em etapas, usar um verbo claro."], TEAL),
        ("Formato ruim", ["Sintoma: difícil de usar.", "Ajuste: pedir tabela, limite de palavras, linguagem."], AMBER),
        ("Critério ausente", ["Sintoma: não sei se está bom.", "Ajuste: listar critérios e exemplos e contraexemplos."], PLUM),
        ("Erro de conteúdo", ["Sintoma: afirmação duvidosa.", "Ajuste: apontar o trecho e conferir em fonte externa."], ROSE),
    ], w=8.6, h=5.0, wrap_at=24, colunas=3)

def f25_sintese():
    fig, ax = canvas(8.6, 3.8, (0, 100), (0, 44))
    xs = [2, 26, 50, 74]
    ts = ["Primeira\nresposta", "Critérios e\nexemplos", "Versões\nrevisadas", "Decisão\ndocente"]
    cs = [AMBER, PLUM, BLUE, TERRA]
    for x, t, c in zip(xs, ts, cs):
        rbox(ax, x, 20, 22, 14, t, fc=c, tc="white", bold=True, fs=8.8, r=3, z=3)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + 22.2, 27), (b - 0.2, 27))
    rbox(ax, 2, 4, 94, 9, "Conferência externa em todas as etapas: fontes, colegas, o próprio conhecimento", fc=SAND, bold=True, fs=8.4, r=3, z=3)
    save(fig, "f25_sintese")

if __name__ == "__main__":
    f25_mapa(); f25_ciclo(); f25_diagnostico(); f25_sintese()
