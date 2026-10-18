from fe import *

def f16_mapa():
    mapa_conceitos("f16_mapa", "IA na\ngraduação", [
        ("Público e contexto", ["trabalho e estudo", "formação profissional"]),
        ("Evidências", ["heterogêneas", "curto prazo"]),
        ("Orientações de uso", ["três níveis", "declaração"]),
        ("Atividades", ["parecer crítico", "recuperação espaçada"]),
        ("Avaliação e integridade", ["processo e oral", "sem detectores"]),
        ("Riscos", ["direitos autorais", "desigualdade"]),
    ], w=8.6, h=6.0, cor=PLUM)

def f16_niveis():
    semaforo("f16_niveis", [
        ("r", "Nível 1: sem IA", "Provas orais, escritas em sala e demonstrações; objetivo: domínio individual."),
        ("a", "Nível 2: IA como apoio declarado", "Estudo, revisão e feedback, com registro do uso e responsabilidade do estudante."),
        ("v", "Nível 3: IA como objeto e parceira de projeto", "Análise crítica e projetos integradores; avalia-se o processo e a verificação.")], w=8.6, wrap_at=95)

def f16_sintese():
    fig, ax = canvas(8.4, 4.2, (0, 100), (0, 50))
    for x, y, t, c in [(4, 28, "Política de uso\n(níveis e declaração)", TERRA), (54, 28, "Atividades alinhadas\n(apoio, objeto, projeto)", TEAL), (4, 4, "Avaliação diversificada\n(processo, oral, em sala)", PLUM), (54, 4, "Verificação\n(fontes primárias)", BLUE)]:
        rbox(ax, x, y, 42, 16, t, fc=c, tc="white", bold=True, fs=8.6, r=3, z=3)
    ax.add_patch(Circle((50, 25), 8.5, fc=SAND, ec=GOLD, lw=2, zorder=5))
    ax.text(50, 25, "formação", ha="center", va="center", fontsize=9, fontweight="bold", zorder=6)
    save(fig, "f16_sintese")

if __name__ == "__main__":
    f16_mapa(); f16_niveis(); f16_sintese()
