from fe import *

def f30_mapa():
    mapa_conceitos("f30_mapa", "Dependência,\ntrabalho e\npolíticas", [
        ("Tipos de dependência", ["cognitiva", "de julgamento", "organizacional"]),
        ("Sinais e salvaguardas", ["autoexame", "tarefas-treino"]),
        ("Trabalho docente", ["tempo reabsorvido", "autonomia"]),
        ("Direito de não usar", ["sem penalidade", "alternativas"]),
        ("Política institucional", ["construção coletiva", "revisão"]),
        ("Riscos", ["vigilância", "intensificação"]),
    ], w=8.6, h=6.0, cor=ROSE)

def f30_espectro():
    espectro("f30_espectro", "Escolha consciente", "Perda de controle", [
        (0.05, "Uso deliberado"), (0.28, "Hábito útil"), (0.5, "Hábito automático"),
        (0.74, "Perda de confiança"), (0.95, "Dependência")], w=8.6, h=3.0, cor_esq=GREEN, cor_dir=TERRA)

def f30_trabalho():
    comparar_cartoes("f30_trabalho", [
        ("A preservar", ["Relação com estudantes e famílias", "Julgamento avaliativo", "Decisões éticas e de acolhimento"], TEAL),
        ("A apoiar, com revisão", ["Rascunhos de materiais", "Ideias de atividades", "Adaptações de formato"], AMBER),
        ("A avaliar para delegar", ["Tarefas repetitivas e administrativas", "Sem dados pessoais", "Com verificação por amostra"], PLUM),
    ], w=8.6, h=4.2, wrap_at=22)

def f30_politica():
    fluxo("f30_politica", [
        ("Escutar", "docentes, estudantes, famílias"), ("Mapear", "usos e riscos reais"),
        ("Redigir", "princípios e regras simples"), ("Testar", "em pequena escala"),
        ("Publicar", "com formação"), ("Revisar", "a cada semestre")], w=8.6, cor="multi", por_linha=3, wrap_at=16)

def f30_sintese():
    fig, ax = canvas(8.4, 4.4, (0, 100), (0, 52))
    for r, c, t in [(24, MID, ""), (17, SAND, ""), (9, ROSE, "")]:
        ax.add_patch(Circle((50, 26), r, fc=c, ec=GRY, lw=1, zorder=1 + (24 - r) / 10))
    ax.text(50, 26, "Docente\nque observa\no próprio uso", ha="center", va="center", fontsize=8, color="white", fontweight="bold", zorder=6)
    ax.text(50, 37.5, "Coletivo que\ndialoga e decide", ha="center", va="center", fontsize=7.4, fontweight="bold", zorder=6)
    ax.text(50, 46.2, "Instituição:\npolítica e formação", ha="center", va="center", fontsize=7.4, fontweight="bold", zorder=6)
    rbox(ax, 2, 20, 20, 12, "Direito de\nnão usar", fc=TEAL, tc="white", bold=True, fs=8, r=2.5, z=5)
    rbox(ax, 78, 20, 20, 12, "Proteção\nde dados", fc=BLUE, tc="white", bold=True, fs=8, r=2.5, z=5)
    arrow(ax, (22.3, 26), (33, 26)); arrow(ax, (77.7, 26), (67, 26))
    save(fig, "f30_sintese")

if __name__ == "__main__":
    f30_mapa(); f30_espectro(); f30_trabalho(); f30_politica(); f30_sintese()
