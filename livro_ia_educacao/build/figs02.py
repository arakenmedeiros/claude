from fe import *

def f02_mapa():
    mapa_conceitos("f02_mapa", "Inteligência\nartificial", [
        ("Definições possíveis", ["agir/pensar como humanos", "agir racionalmente"]),
        ("Automação por regras", ["calculadora, planilha", "previsível"]),
        ("Aprender com dados", ["padrões a partir de exemplos", "pode errar"]),
        ("Família da IA", ["IA ⊃ aprendizado de máquina", "⊃ profundo ⊃ generativa"]),
        ("IA na educação", ["previsão e avaliação", "adaptação e tutoria"]),
        ("Mitos comuns", ["\"entende\"", "\"é neutra\""]),
    ], w=8.6, h=6.0, cor=PLUM)

def f02_espectro():
    espectro("f02_espectro", "Regras escritas por pessoas", "Padrões aprendidos de dados",
        [(0.03, "Calculadora"), (0.2, "Semáforo\ncom tempo fixo"), (0.43, "Corretor\nortográfico"), (0.64, "Filtro de spam"), (0.82, "Reconhecimento\nde voz"), (0.97, "Assistente\nconversacional")],
        w=8.6, h=3.6, cor_esq=TEAL, cor_dir=PLUM)

def f02_conjuntos():
    fig, ax = canvas(8.4, 5.0, (0, 100), (0, 58))
    rbox(ax, 2, 16, 96, 40, "", fc=MID, ec=GRY, lw=1.4, r=2.5, z=1)
    ax.text(4.5, 53, "Inteligência artificial (campo amplo)", ha="left", va="center", fontsize=9.2, fontweight="bold", color=INK, zorder=6)
    rbox(ax, 5, 19, 25, 29, "", fc=SAND, ec=GOLD, lw=1.4, r=2, z=2)
    ax.text(17.5, 36, wrap("Sistemas baseados em regras", 18), ha="center", va="center", fontsize=8.4, fontweight="bold", zorder=6, linespacing=1.1)
    ax.text(17.5, 26, wrap("ex.: sistemas especialistas", 18), ha="center", va="center", fontsize=7.6, zorder=6, color=GRY)
    rbox(ax, 33, 19, 63, 29, "", fc="#F8D9B8", ec=TERRA, lw=1.4, r=2, z=2)
    ax.text(64.5, 44.5, "Aprendizado de máquina (família de métodos)", ha="center", va="center", fontsize=8.4, fontweight="bold", zorder=6)
    rbox(ax, 36, 22, 22, 18, "", fc="#FBE9D3", ec=TERRA, lw=1.0, r=1.5, z=3)
    ax.text(47, 31, wrap("Outros métodos de aprendizado", 17), ha="center", va="center", fontsize=7.8, zorder=6, linespacing=1.1)
    rbox(ax, 61, 22, 32, 18, "", fc=AMBER, ec=GOLD, lw=1.2, r=1.5, z=3)
    ax.text(77, 31, wrap("Aprendizado profundo (redes neurais)", 24), ha="center", va="center", fontsize=8.2, fontweight="bold", zorder=6, linespacing=1.1)
    rbox(ax, 2, 1, 96, 11, "", fc="#F3E8F4", ec=PLUM, lw=1.5, r=2.5, z=1)
    ax.text(50, 6.5, wrap("Capacidade de gerar conteúdo (texto, imagem, áudio, código): pode ser implementada por métodos diferentes. Os assistentes generativos atuais, como os modelos de linguagem de grande porte, usam em geral aprendizado profundo.", 92), ha="center", va="center", fontsize=7.8, zorder=6, linespacing=1.2)
    arrow(ax, (77, 12.4), (77, 21.6), c=PLUM, lw=1.8, z=9)
    save(fig, "f02_conjuntos")

def f02_usos():
    comparar_cartoes("f02_usos", [
        ("Perfil e previsão", ["Estimar risco de evasão", "Prever dificuldades", "Foco: gestão e professor"], TERRA),
        ("Avaliação", ["Correção assistida", "Análise de respostas", "Foco: professor e estudante"], TEAL),
        ("Adaptação e personalização", ["Sequências de exercícios", "Recomendação de leituras", "Foco: estudante"], PLUM),
        ("Tutores inteligentes", ["Orientação passo a passo", "Dicas e retorno imediato", "Foco: estudante"], BLUE)],
        w=8.6, h=2.9, wrap_at=22)

def f02_sintese():
    fig, ax = canvas(6.4, 5.4, (0, 100), (0, 84))
    perg = [("Alguém escreveu o passo a passo?", "Sim", "Automação por regras", TEAL, 70), ("Aprendeu padrões a partir de exemplos?", "Sim", "Aprendizado de máquina", AMBER, 44), ("Cria conteúdo novo (texto, imagem, áudio)?", "Sim", "IA generativa", TERRA, 18)]
    for i, (q, s, r, c, y) in enumerate(perg):
        rbox(ax, 4, y - 6, 48, 12, q, fc=WHITE, ec=c, lw=2, r=3, fs=8.2, wrap_at=30, z=3)
        arrow(ax, (52, y), (63, y), c, lw=2); label(ax, 57.5, y + 3.2, s, 7.5, c, True)
        rbox(ax, 63, y - 5, 35, 10, r, fc=c, tc="white", bold=True, fs=7.4, r=3, wrap_at=22, z=3)
        if i < 2:
            arrow(ax, (28, y - 6.2), (28, perg[i + 1][4] + 6.2), GRY, lw=1.6); label(ax, 33, (y + perg[i + 1][4]) / 2, "Não", 7.5, GRY, True, ha="left")
    ax.text(50, 4, "Lembre: muitos sistemas combinam regras e aprendizado.", ha="center", fontsize=7.8, style="italic", color=GRY)
    save(fig, "f02_sintese")

if __name__ == "__main__":
    f02_mapa(); f02_espectro(); f02_conjuntos(); f02_usos(); f02_sintese()
