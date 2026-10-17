from fe import *

def f08_mapa():
    mapa_conceitos("f08_mapa", "Autorregulação,\nmediação e\nfeedback", [
        ("Ciclo de autorregulação", ["antecipação", "desempenho, reflexão"]),
        ("Metacognição", ["conhecimento", "regulação"]),
        ("Mediação docente", ["antes e durante", "depois e ao final"]),
        ("Três perguntas", ["meta, situação", "próximo passo"]),
        ("Feedback gerado por IA", ["potencial e riscos", "não substitui sozinho"]),
        ("Duas camadas", ["IA ou par primeiro", "devolutiva do professor"]),
    ], w=8.6, h=6.0, cor=PLUM)

def f08_ciclo():
    fig, ax = canvas(8.6, 4.6, (-58, 58), (-30, 30))
    fases = [("1. Antecipação", "metas e planejamento", TERRA, (0, 22)), ("2. Desempenho", "execução e automonitoramento", TEAL, (34, -15)), ("3. Autorreflexão", "avaliação e ajuste", PLUM, (-34, -15))]
    for t, d, c, (x, y) in fases:
        rbox(ax, x - 14, y - 5.5, 28, 11, "", fc=WHITE, ec=c, lw=2.2, r=3, z=3)
        ax.text(x, y + 2, t, ha="center", va="center", fontsize=8.6, fontweight="bold", color=c, zorder=5)
        ax.text(x, y - 2.3, d, ha="center", va="center", fontsize=7, zorder=5)
    arrow(ax, (14.5, 20), (30, -8.5), GRY, lw=1.8, rad=-0.3); arrow(ax, (20, -15), (-20, -15), GRY, lw=1.8, rad=-0.0); arrow(ax, (-30, -8.5), (-14.5, 20), GRY, lw=1.8, rad=-0.3)
    apoio = [(-30, 26, "Apoio possível: perguntas de planejamento", TERRA), (46, 3, "Apoio possível: dicas graduais", TEAL), (-46, 3, "Apoio possível: comparar autoavaliação com critérios", PLUM)]
    ax.text(0, -26.5, "Risco em qualquer fase: a IA decidir no lugar do estudante.", ha="center", fontsize=8, color=ROSE, fontweight="bold")
    for x, y, t, c in [(28, 26, "IA pode sugerir\nperguntas de planejamento", TERRA), (43, 1, "IA pode dar\ndicas graduais", TEAL), (-43, 1, "IA pode comparar a\nautoavaliação com critérios", PLUM)]:
        rbox(ax, x - 12, y - 4, 24, 8, t, fc=SAND, ec=c, fs=6.6, r=2, z=3, lw=1.2)
    save(fig, "f08_ciclo")

def f08_feedback():
    fig, ax = canvas(8.6, 4.2, (0, 100), (0, 49))
    qs = [("Para onde estou indo?", "meta e critério", TERRA, 38), ("Como estou indo?", "situação atual", TEAL, 24), ("Para onde vou agora?", "próximo passo", PLUM, 10)]
    for t, d, c, y in qs:
        rbox(ax, 2, y - 4.5, 40, 9, "", fc=WHITE, ec=c, lw=2, r=2.5, z=3)
        ax.text(22, y + 1.4, t, ha="center", va="center", fontsize=8.6, fontweight="bold", color=c, zorder=5)
        ax.text(22, y - 2.3, d, ha="center", va="center", fontsize=7.4, zorder=5)
    niv = [("Tarefa", "o produto: específico ajuda", TEAL, 42), ("Processo", "estratégias usadas", BLUE, 30.5), ("Autorregulação", "monitorar o próprio trabalho", PLUM, 19), ("Pessoa", "elogios genéricos: pouco informam", GRY, 7.5)]
    for t, d, c, y in niv:
        rbox(ax, 50, y - 4.8, 48, 9.6, "", fc=c, r=2, z=3, alpha=0.92)
        ax.text(53, y + 1.3, t, ha="left", va="center", fontsize=8.4, fontweight="bold", color="white", zorder=5)
        ax.text(53, y - 2.5, d, ha="left", va="center", fontsize=7.2, color="white", zorder=5)
    label(ax, 22, 46.2, "Três perguntas", 9, INK, True); label(ax, 74, 49, "Quatro níveis", 9, INK, True)
    ax.set_ylim(0, 51)
    save(fig, "f08_feedback")

def f08_riscos():
    comparar_cartoes("f08_riscos", [
        ("Qualidades desejáveis", ["Alinhado aos critérios", "Específico, com evidências do texto", "Orienta o próximo passo", "Respeita a autoria", "Tom acolhedor"], GREEN),
        ("Riscos", ["Genérico ou elogioso demais", "Incorreto ou inventado", "Descontextualizado", "Reescreve em vez de orientar", "Tom inadequado e dados expostos"], ROSE)],
        w=8.6, h=4.0, wrap_at=38)

def f08_sintese():
    ciclo("f08_sintese", [("Rascunho", "sem IA"), ("Autoavaliação", "com a rubrica"), ("Feedback inicial", "IA ou colega"), ("Decisões", "aceito, rejeito,\nadapto"), ("Reescrita e\ndevolutiva", "professor comenta\nas decisões")], centro="Estudante", w=7.6, h=5.2, wrap_at=14)

if __name__ == "__main__":
    f08_mapa(); f08_ciclo(); f08_feedback(); f08_riscos(); f08_sintese()
