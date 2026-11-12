from fe import *

def f10_mapa():
    mapa_conceitos("f10_mapa", "Decidir sobre\no uso de IA", [
        ("Seis critérios", ["objetivo, evidências", "riscos, alternativas"]),
        ("Quando agrega valor", ["acesso e andaimes", "tempo do professor"]),
        ("Quando prejudica", ["elimina o esforço", "expõe dados"]),
        ("Fluxograma", ["cinco perguntas", "usar, adaptar, dispensar"]),
        ("Classificação de risco", ["baixo, moderado, alto", "e excessivo (vedado)"]),
        ("Plano A, B e C", ["com IA, poucos recursos", "sem IA"]),
    ], w=8.6, h=6.0, cor=BLUE)

def f10_decisao():
    nos = {
        "i": (3, 80, 50, 6.5, "Tarefa que estou planejando", "i"),
        "q1": (3, 67.5, 50, 8, "1. O objetivo está claro?", "q"),
        "a1": (62, 67.5, 36, 8, "Defina o objetivo antes de decidir", "a"),
        "q2": (3, 54.5, 50, 8, "2. O estudante precisa fazer sozinho o esforço que a IA faria?", "q"),
        "x1": (62, 54.5, 36, 8, "Dispensar a IA nesta etapa", "x"),
        "q3": (3, 41.5, 50, 8, "3. Há dados pessoais ou decisões com consequências sobre pessoas?", "q"),
        "x2": (62, 41.5, 36, 8, "Dispensar ou usar só versão anonimizada", "x"),
        "q4": (3, 28.5, 50, 8, "4. Consigo verificar e há alternativa sem IA?", "q"),
        "x3": (62, 28.5, 36, 8, "Dispensar ou adiar o uso", "x"),
        "q5": (3, 15.5, 50, 8, "5. Idade, normas e condições permitem uso com estudantes?", "q"),
        "f1": (62, 15.5, 36, 8, "Usar com mediação, salvaguardas e registro", "f"),
        "f2": (3, 2.5, 50, 8, "Usar só na preparação, com revisão", "f"),
    }
    lig = [("i", "b", "q1", "t", "", 0), ("q1", "b", "q2", "t", "Sim", 0), ("q1", "r", "a1", "l", "Não", 0),
           ("q2", "r", "x1", "l", "Sim", 0), ("q2", "b", "q3", "t", "Não", 0),
           ("q3", "r", "x2", "l", "Sim", 0), ("q3", "b", "q4", "t", "Não", 0),
           ("q4", "r", "x3", "l", "Não", 0), ("q4", "b", "q5", "t", "Sim", 0),
           ("q5", "r", "f1", "l", "Sim", 0), ("q5", "b", "f2", "t", "Não", 0)]
    decisao("f10_decisao", nos, lig, w=7.0, h=6.2, fs=8.4)

def semaforo4(name, linhas, w=8.6, fs=8.2, wrap_at=95):
    """Quatro níveis: v (baixo), a (moderado), r (alto), x (excessivo/vedado)."""
    h = 1.6 + 1.0 * len(linhas) * 1.05
    fig, ax = canvas(w, h, (0, 100), (0, 100 * h / w))
    H = 100 * h / w
    cmap = {"v": (GREEN, "#EAF3E3"), "a": (AMBER, "#FDF1D8"), "r": (ROSE, "#FADDE5"), "x": (PLUM, "#EBE0EE")}
    lh = (H - 3) / len(linhas)
    for i, (k, t, d) in enumerate(linhas):
        c, f = cmap[k]
        y = H - 1.5 - (i + 1) * lh
        rbox(ax, 2, y + 0.6, 96, lh - 1.2, "", fc=f, ec=c, lw=1.5, r=2, z=2)
        ax.add_patch(Circle((8, y + lh / 2), min(3.6, lh * 0.32), fc=c, ec="none", zorder=4))
        ax.text(14, y + lh / 2 + (1.1 if d else 0), t, ha="left", va="center", fontsize=fs + 0.3, fontweight="bold", color=INK, zorder=5)
        if d: ax.text(14, y + lh / 2 - 2.0, wrap(d, wrap_at), ha="left", va="center", fontsize=fs - 1.4, zorder=5, linespacing=1.1)
    save(fig, name)

def f10_risco():
    semaforo4("f10_risco", [
        ("v", "Baixo risco", "Apoio instrumental ao professor (organizar e adaptar materiais, revisão textual) e acessibilidade simples, sem perfilização."),
        ("a", "Risco moderado", "Interação contínua com estudantes, tutoria, personalização de percursos e feedback formativo sem efeito na nota."),
        ("r", "Alto risco", "Efeitos sobre avaliação, certificação ou direitos: proctoring, predição de evasão, detectores de IA, pré-correção."),
        ("x", "Excessivo (vedado)", "Pontuação social, reconhecimento de emoções, perfilização, decisão só automatizada, nota só por IA.")], w=8.6, wrap_at=95)

def f10_plano():
    comparar_cartoes("f10_plano", [
        ("Plano A: com IA", ["Quando as condições permitem", "Mediação e salvaguardas", "Registro de decisões"], TERRA),
        ("Plano B: poucos recursos", ["IA só na preparação", "Aula sem tecnologia ou offline", "Mesmo objetivo"], TEAL),
        ("Plano C: sem IA", ["Papel, conversa e materiais da escola", "Mesmo objetivo", "Sempre disponível"], PLUM)],
        w=8.6, h=3.0, wrap_at=24)

def f10_sintese():
    fig, ax = canvas(8.2, 4.4, (0, 100), (0, 54))
    ax.plot([50, 50], [8, 36], c=INK, lw=3)
    ax.plot([14, 86], [38, 38], c=INK, lw=3)
    ax.add_patch(Polygon([[44, 8], [56, 8], [50, 14]], fc=INK))
    for x, t, d, c in [(8, "Valor", "valor pedagógico,\nacesso, tempo do\nprofessor", GREEN), (66, "Riscos", "riscos, custos,\ndependência", ROSE)]:
        ax.plot([x + 8, x + 8], [38, 31], c=INK, lw=1.4)
        rbox(ax, x - 1, 15, 26, 16, "", fc=c, r=2, z=3, alpha=0.95)
        ax.text(x + 12, 27, t, ha="center", va="center", fontsize=9.5, fontweight="bold", color="white", zorder=5)
        ax.text(x + 12, 20, d, ha="center", va="center", fontsize=7.4, color="white", zorder=5, linespacing=1.15)
    rbox(ax, 28, 0.5, 44, 6, "Critérios: objetivo, evidência e alternativas", fc=TERRA, tc="white", bold=True, fs=7.8, r=2, z=3)
    label(ax, 50, 46, "A decisão é do professor", 10, INK, True)
    save(fig, "f10_sintese")

if __name__ == "__main__":
    f10_mapa(); f10_decisao(); f10_risco(); f10_plano(); f10_sintese()
