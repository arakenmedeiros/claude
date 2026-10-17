from fe import *

def f03_mapa():
    mapa_conceitos("f03_mapa", "Como a IA\ngenerativa\nfunciona", [
        ("Dados de treinamento", ["textos de várias fontes", "lacunas e vieses"]),
        ("Treinamento em etapas", ["prever o próximo token", "ajuste e feedback"]),
        ("Tokens e probabilidades", ["pedaços de texto", "sorteio entre opções"]),
        ("Geração e variabilidade", ["temperatura", "janela de contexto"]),
        ("Multimodais e buscas", ["texto, imagem, áudio", "RAG e documentos"]),
        ("Consequências docentes", ["verificar sempre", "decisão é humana"]),
    ], w=8.6, h=6.0, cor=BLUE)

def f03_tokens():
    fig, ax = canvas(8.2, 2.6, (0, 100), (0, 31))
    toks = ["A", " profess", "ora", " corrig", "iu", " as", " prov", "as", "."]
    ids = [32, 4510, 988, 7312, 117, 302, 2276, 54, 13]
    cols = [TERRA, AMBER, TEAL, PLUM, BLUE, GREEN, ROSE, CORAL, GOLD]
    x = 4
    for t, i, c in zip(toks, ids, cols):
        w = 3.2 + 1.45 * len(t.strip())
        rbox(ax, x, 15, w, 9, "", fc=c, r=1.2, alpha=0.9, z=3)
        ax.text(x + w / 2, 19.5, t.replace(" ", "·") if t.startswith(" ") else t, ha="center", va="center", color="white", fontsize=9, fontweight="bold", zorder=5)
        ax.text(x + w / 2, 10, str(i), ha="center", va="center", fontsize=7.5, color=INK, zorder=5)
        ax.plot([x + w / 2, x + w / 2], [14.6, 11.8], c=GRY, lw=0.8)
        x += w + 1.2
    label(ax, 4, 28, "Texto: \"A professora corrigiu as provas.\"", 8.6, INK, True, ha="left")
    label(ax, 4, 3.6, "Números meramente ilustrativos. O ponto · indica um espaço no início do token.", 7.2, GRY, False, ha="left", it=True)
    save(fig, "f03_tokens")

def f03_pipeline():
    fluxo("f03_pipeline", [("Dados", "textos de muitas fontes"), ("Pré-treino", "prever o próximo token"), ("Ajuste", "seguir instruções"), ("Feedback", "humano e regras de segurança"), ("Assistente", "conversa com o usuário")],
          w=8.6, cor="multi", wrap_at=14, h=2.9)

def f03_proxima():
    fig, ax = canvas(8.2, 3.6, (0, 100), (0, 43))
    label(ax, 3, 39, "O professor entrou na sala e abriu o ______", 10, INK, True, ha="left")
    ops = [("livro", 0.38), ("caderno", 0.27), ("computador", 0.19), ("cadeado", 0.01), ("oceano", 0.002)]
    for i, (p, v) in enumerate(ops):
        y = 30 - i * 5.6
        ax.text(3, y, p, ha="left", va="center", fontsize=9)
        w = 40 * v / 0.38
        rbox(ax, 20, y - 1.9, max(w, 0.8), 3.8, "", fc=[TERRA, AMBER, TEAL, GRY, GRY][i], r=0.8, z=3)
        ax.text(20 + max(w, 0.8) + 1, y, "mais provável" if i == 0 else ("muito improvável" if i >= 3 else ""), fontsize=7.4, va="center", color=GRY)
    rbox(ax, 66, 8, 32, 26, "", fc=SAND, ec=GOLD, r=2, z=2)
    ax.text(82, 21, wrap("O sistema sorteia entre as opções de acordo com as probabilidades. Por isso, a resposta pode variar de uma vez para outra.", 28), ha="center", va="center", fontsize=7.8, zorder=5, linespacing=1.2)
    label(ax, 50, 1.5, "Probabilidades inventadas para fins didáticos; não representam nenhum modelo real.", 7.2, GRY, it=True)
    save(fig, "f03_proxima")

def f03_multimodal():
    comparar_cartoes("f03_multimodal", [
        ("Texto", ["Explicações e exemplos", "Rascunhos de perguntas", "Verificar: fatos, datas, fontes"], TERRA),
        ("Imagem", ["Ilustrações de apoio", "Cartazes e cenas", "Verificar: erros e estereótipos"], TEAL),
        ("Áudio", ["Leitura em voz alta", "Transcrição de aulas", "Verificar: fidelidade e privacidade"], PLUM),
        ("Multimodal", ["Descrever uma figura", "Explicar foto de exercício", "Verificar: se descreve o que há"], BLUE)],
        w=8.6, h=3.0, wrap_at=22)

def f03_sintese():
    fluxo("f03_sintese", [("Dados", "o que o modelo viu"), ("Treinamento", "padrões de linguagem"), ("Prompt", "o que você pede"), ("Resposta plausível", "fluente, mas a verificar"), ("Julgamento docente", "validar, adaptar, decidir")],
          w=8.6, cor="multi", wrap_at=14, h=2.9)

if __name__ == "__main__":
    f03_mapa(); f03_tokens(); f03_pipeline(); f03_proxima(); f03_multimodal(); f03_sintese()
