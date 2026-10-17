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
    fig, ax = canvas(8.2, 4.8, (0, 100), (-2, 56))
    cx, cy = 34, 29
    cores = [MID, "#F8D9B8", "#F2A93B", TERRA]
    rs = [27, 21, 14.5, 8]
    nomes = ["Inteligência artificial", "Aprendizado de máquina", "Aprendizado profundo", "IA generativa\ne LLMs"]
    for r, c, n, i in zip(rs, cores, nomes, range(4)):
        ax.add_patch(Circle((cx, cy - (27 - r)), r, fc=c, ec="white", lw=2, zorder=2 + i))
        ax.text(cx, cy - (27 - r) + r - 3.6 if i < 3 else cy - (27 - r), n, ha="center", va="center", fontsize=8.2 if i < 3 else 7.6, fontweight="bold", color=INK if i < 3 else "white", zorder=8, linespacing=1.05)
    rbox(ax, 66, 34, 32, 18, "", fc=SAND, ec=GOLD, lw=1.6, r=2, z=2)
    ax.text(82, 43, wrap("Há IA sem aprendizado de máquina: os sistemas especialistas baseados em regras.", 30), ha="center", va="center", fontsize=7.6, zorder=5, linespacing=1.2)
    rbox(ax, 66, 8, 32, 18, "", fc="#EAF6F4", ec=TEAL, lw=1.6, r=2, z=2)
    ax.text(82, 17, wrap("Imagine bonecas russas: cada conceito está dentro do anterior.", 30), ha="center", va="center", fontsize=7.6, zorder=5, linespacing=1.2)
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
