from fe import *

def f01_mapa():
    mapa_conceitos("f01_mapa", "Educação,\ntecnologia\ne sociedade",
        [("História das tecnologias na escola", ["escrita, imprensa, rádio, TV", "computador e internet"]),
         ("Mudanças sociais", ["informação em abundância", "trabalho e plataformas"]),
         ("Competências nos documentos", ["BNCC: cultura digital", "Computação na Educação Básica"]),
         ("A chegada da IA generativa", ["fim de 2022", "reações diversas"]),
         ("Três posturas possíveis", ["entusiasmo acrítico", "rejeição automática", "deliberação"]),
         ("Decisão pedagógica", ["objetivo primeiro", "usar, adaptar ou dispensar"])],
        w=8.6, h=6.0, cor=TERRA)

def f01_linha():
    fig, ax = canvas(8.6, 3.6, (0, 100), (0, 42))
    marcos = [("Escrita", "guardar o saber\nfora da memória", TERRA), ("Imprensa", "livros para\nmuito mais gente", AMBER), ("Rádio e TV", "aulas a lugares\ndistantes", TEAL),
              ("Computador", "tecnologia na\nsala de aula", BLUE), ("Internet", "acesso global à\ninformação", PLUM), ("Celular e\nplataformas", "dados, redes e\nalgoritmos", ROSE), ("IA generativa", "conversa e geração\nde conteúdo", GREEN)]
    ax.plot([3, 97], [21, 21], c=MID, lw=6, solid_capstyle="round", zorder=1)
    for i, (t, d, c) in enumerate(marcos):
        x = 8 + i * 14.0
        ax.add_patch(Circle((x, 21), 3.0, fc=c, ec="white", lw=2.5, zorder=3))
        ax.text(x, 21, str(i + 1), ha="center", va="center", color="white", fontsize=8, fontweight="bold", zorder=4)
        up = i % 2 == 0
        ty = 31.5 if up else 10.5
        ax.plot([x, x], [21 + (3.2 if up else -3.2), ty + (-4.2 if up else 4.2)], c=c, lw=1.5, zorder=2)
        rbox(ax, x - 7.2, ty - 4.2, 14.4, 8.4, "", fc=WHITE, ec=c, lw=1.8, r=1.8, z=2)
        ax.text(x, ty + 1.9, t, ha="center", va="center", fontsize=7.6, fontweight="bold", color=c, linespacing=1.0, zorder=5)
        ax.text(x, ty - 1.9, d, ha="center", va="center", fontsize=5.6, linespacing=1.05, zorder=5)
    ax.text(50, 1.2, "Sem escala temporal: a figura mostra a sequência, não as datas exatas.", ha="center", fontsize=7, color=GRY, style="italic")
    save(fig, "f01_linha")

def f01_lentes():
    comparar_cartoes("f01_lentes", [
        ("Entusiasmo acrítico", ["Acredita que a tecnologia melhora a aprendizagem por si só", "Repete slogans e promessas", "Risco: adotar sem evidência e sem olhar quem fica de fora"], AMBER),
        ("Rejeição automática", ["Acredita que a tecnologia prejudica a aprendizagem por si só", "Recorre à proibição", "Risco: ignorar usos possíveis e deixar estudantes sem orientação"], ROSE),
        ("Deliberação fundamentada", ["Parte do objetivo de aprendizagem", "Analisa evidências, riscos e alternativas", "Pode concluir: usar, adaptar ou dispensar"], TEAL)],
        w=8.6, h=3.3, wrap_at=26)

def f01_sintese():
    ciclo("f01_sintese", [("Objetivo de\naprendizagem", "o que os estudantes\nprecisam aprender"), ("Análise do\ncontexto", "turma, recursos,\ntempo, riscos"),
                          ("Decisão", "usar, adaptar\nou dispensar a IA"), ("Avaliação do\nresultado", "o que de fato\nfoi aprendido")], centro="Professor", w=7.2, h=5.0)

if __name__ == "__main__":
    f01_mapa(); f01_linha(); f01_lentes(); f01_sintese()
