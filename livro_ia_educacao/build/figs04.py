from fe import *

def f04_mapa():
    mapa_conceitos("f04_mapa", "Erros, vieses\ne limites", [
        ("Tipos de erro", ["factual, raciocínio", "bajulação, desatualização"]),
        ("Fontes inventadas", ["referências inexistentes", "taxas variam"]),
        ("Por que acontece", ["texto provável", "forma sem conteúdo"]),
        ("Vieses", ["dados, projeto, uso", "média esconde grupos"]),
        ("Três filtros", ["fluência", "correção e valor pedagógico"]),
        ("Verificação", ["marcar, procurar, refazer", "avaliar, registrar"]),
    ], w=8.6, h=6.0, cor=ROSE)

def f04_tipos():
    comparar_cartoes("f04_tipos", [
        ("Erro factual", ["Data ou nome incorreto", "Ex.: ano errado de um evento histórico"], TERRA),
        ("Fonte inventada", ["Referência ou lei que não existe", "Ex.: artigo com autor e título plausíveis"], PLUM),
        ("Raciocínio ou cálculo", ["Passos lógicos com falha", "Ex.: conta de fração com resultado errado"], BLUE),
        ("Desatualização", ["Informação anterior à data de corte", "Ex.: norma já revogada"], GOLD),
        ("Concordância excessiva", ["Aceita a premissa errada do usuário", "Ex.: confirma uma ideia equivocada"], TEAL),
        ("Viés e estereótipo", ["Reflete desigualdades dos dados", "Ex.: personagens sempre nos mesmos papéis"], ROSE)],
        w=8.6, h=5.3, wrap_at=24, colunas=3)

def f04_vies():
    fig, ax = canvas(8.6, 3.0, (0, 100), (0, 35))
    blocos = [("Dados", "quem aparece e quem\nnão aparece nos textos\ne nas imagens", TERRA), ("Projeto", "o que se decide medir,\nrotular e otimizar", TEAL), ("Uso", "como as pessoas aplicam\nos resultados e\nem que decisões", PLUM)]
    for i, (t, d, c) in enumerate(blocos):
        x = 3 + i * 33
        rbox(ax, x, 8, 28, 22, "", fc=WHITE, ec=c, lw=2, r=3, z=2)
        rbox(ax, x, 23.5, 28, 6.5, t, fc=c, tc="white", bold=True, fs=9.5, r=3, z=3)
        ax.text(x + 14, 15.5, d, ha="center", va="center", fontsize=7.8, linespacing=1.2, zorder=5)
        if i < 2: arrow(ax, (x + 28.3, 19), (x + 32.7, 19), INK, lw=1.8)
    arrow(ax, (86, 7.2), (17, 7.2), GRY, lw=1.6, rad=-0.0, ls="--")
    ax.text(51, 3.2, "os efeitos do uso retornam aos dados e reforçam o padrão (ciclo)", ha="center", fontsize=7.8, style="italic", color=GRY)
    save(fig, "f04_vies")

def f04_filtros():
    fig, ax = canvas(7.6, 4.4, (0, 100), (0, 58))
    est = [("Fluência", "está bem escrita?", AMBER, 94), ("Correção", "está certa e verificável?", TEAL, 82), ("Valor pedagógico", "serve ao objetivo, à idade e ao nível?", PLUM, 70)]
    for i, (t, d, c, w) in enumerate(est):
        y = 43 - i * 14
        ax.add_patch(Polygon([[50 - w / 2, y + 12], [50 + w / 2, y + 12], [50 + (w - 8) / 2, y], [50 - (w - 8) / 2, y]], fc=c, ec="white", lw=2))
        ax.text(50, y + 8, t, ha="center", va="center", fontsize=9.5, fontweight="bold", color="white")
        ax.text(50, y + 3.6, d, ha="center", va="center", fontsize=7.8, color="white")
    arrow(ax, (50, 3.4), (50, -0.8), INK, lw=2)
    rbox(ax, 30, -7.6, 40, 6, "Segue para a sala de aula", fc=GREEN, tc="white", bold=True, fs=8.5, r=2, z=3)
    ax.set_ylim(-9, 56)
    save(fig, "f04_filtros")

def f04_verifica():
    fluxo("f04_verifica", [("Marcar", "datas, nomes, números, referências"), ("Procurar", "fontes independentes e confiáveis"), ("Refazer", "cálculos e exemplos"), ("Avaliar", "idade, nível, diversidade"), ("Registrar", "o que mudou e por quê")], w=8.6, cor="multi", wrap_at=15, h=2.9)

def f04_sintese():
    fig, ax = canvas(8.6, 3.4, (0, 100), (0, 40))
    rbox(ax, 2, 14, 17, 12, "Resposta\nde IA", fc=SAND, ec=GOLD, lw=1.8, fs=8.8, r=2.5, bold=True, z=3)
    xs = [27, 43, 59]
    for x, (t, c) in zip(xs, [("Fluência", AMBER), ("Correção", TEAL), ("Valor\npedagógico", PLUM)]):
        rbox(ax, x, 14, 13.5, 12, t, fc=c, tc="white", bold=True, fs=7.6, r=2.5, z=3)
    for a, b in [(19.3, 26.7), (40.7, 42.7), (56.7, 58.7)]:
        arrow(ax, (a, 20), (b, 20), INK, lw=1.5)
    ax.add_patch(Circle((81, 21), 5.2, fc="none", ec=INK, lw=2.2)); ax.plot([85, 89], [17, 13], c=INK, lw=3)
    ax.text(81, 21, "✓", ha="center", va="center", fontsize=13, color=GREEN, fontweight="bold")
    arrow(ax, (72.8, 20), (75.5, 20), INK, lw=1.5)
    label(ax, 81, 7.2, "Verificação", 8, INK, True)
    rbox(ax, 28, 31, 42, 6.6, "Professor decide: usar, adaptar ou descartar", fc=TERRA, tc="white", bold=True, fs=7.2, r=2.5, z=3)
    arrow(ax, (81, 27), (70.5, 33.6), TERRA, lw=1.5, rad=0.2)
    save(fig, "f04_sintese")

if __name__ == "__main__":
    f04_mapa(); f04_tipos(); f04_vies(); f04_filtros(); f04_verifica(); f04_sintese()
