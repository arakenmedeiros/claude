from fe import *

def f29_mapa():
    mapa_conceitos("f29_mapa", "Privacidade, autoria\ne integridade", [
        ("Dados pessoais", ["LGPD", "minimização"]),
        ("Termos de serviço", ["retenção", "treinamento"]),
        ("Direitos autorais", ["Lei 9.610/1998", "licenças abertas"]),
        ("Autoria e declaração", ["IA não é autora", "modelo de declaração"]),
        ("Integridade", ["processo e oral", "sem detector isolado"]),
        ("Políticas", ["clareza por tarefa", "procedimento justo"]),
    ], w=8.6, h=6.0, cor=ROSE)

def f29_dados():
    semaforo("f29_dados", [
        ("v", "Pode usar", "Conteúdo público ou de licença aberta, planos genéricos, casos fictícios, marcadores sem dados reais."),
        ("a", "Com cuidado e aprovação", "Dados agregados ou anonimizados, em ferramenta aprovada pela instituição, com termos conferidos."),
        ("r", "Não inserir", "Nomes com notas, laudos, imagens e vozes de estudantes, dados de famílias, manuscritos e obras sem autorização."),
    ], w=8.6, wrap_at=90)

def f29_sintese():
    fig, ax = canvas(8.6, 4.2, (0, 100), (0, 49))
    xs = [2, 26, 50, 74]
    ts = ["Há dados\npessoais?", "Há obras de\nterceiros?", "Quem responde\npela autoria?", "O uso foi\ndeclarado?"]
    cs = [TERRA, BLUE, PLUM, TEAL]
    for x, t, c in zip(xs, ts, cs):
        rbox(ax, x, 26, 22, 14, t, fc=c, tc="white", bold=True, fs=8.6, r=3, z=3)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + 22.2, 33), (b - 0.2, 33))
    for x, t, c in [(2, "Dispensar", ROSE), (36, "Adaptar", AMBER), (70, "Usar com cuidado", GREEN)]:
        rbox(ax, x, 4, 26, 11, t, fc=c, tc="white", bold=True, fs=9, r=3, z=3)
    ax.text(50, 20, "Se alguma resposta for \"não sei\", pare e verifique", ha="center", va="center", fontsize=8.4, style="italic", color=INK)
    save(fig, "f29_sintese")

if __name__ == "__main__":
    f29_mapa(); f29_dados(); f29_sintese()
