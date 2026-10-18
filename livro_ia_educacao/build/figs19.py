from fe import *

def f19_mapa():
    mapa_conceitos("f19_mapa", "Linguagens e\nhumanidades com IA", [
        ("Leitura", ["primeiro tentar", "andaime que se retira"]),
        ("Escrita como processo", ["planejar", "rascunhar", "revisar"]),
        ("Feedback em 3 camadas", ["forma", "organização", "sentido e autoria"]),
        ("Línguas", ["variedades", "conversação", "Libras e comunidades"]),
        ("Artes", ["fazer e fruir", "autoria e direitos"]),
        ("Humanidades", ["fontes", "leitura lateral", "quem falta?"]),
    ], w=8.6, h=6.0, cor=GOLD)

def f19_espectro():
    espectro("f19_espectro", "autoria do estudante", "autoria do sistema", [
        (0.02, "ideias e texto próprios"),
        (0.30, "sugestões de forma aceitas e conferidas"),
        (0.62, "trechos gerados e editados"),
        (0.97, "texto integralmente gerado"),
    ], w=8.6, h=3.2)

def f19_processo():
    fluxo("f19_processo", [
        ("Planejar", "estudante: ideias e tese; IA só depois"),
        ("Rascunhar", "estudante, sem apoio; evidência de autoria"),
        ("Revisar sentido", "perguntas de colegas, do professor ou da IA"),
        ("Editar forma", "IA indica; estudante corrige"),
        ("Publicar", "nota de processo e declaração"),
    ], w=8.6, cor="multi", wrap_at=15, por_linha=5, h=2.9)

def f19_sintese():
    fig, ax = canvas(8.6, 4.2, (0, 100), (0, 48))
    rbox(ax, 30, 15, 40, 18, "Processo do\nestudante", fc=GOLD, tc="white", bold=True, fs=10, r=5, z=3)
    for x, t in [(2, "Ler"), (74, "Escrever")]:
        rbox(ax, x, 36, 24, 8, t, fc=TEAL, tc="white", bold=True, fs=9, r=3, z=3)
    for x, t in [(2, "Criar"), (74, "Interpretar")]:
        rbox(ax, x, 4, 24, 8, t, fc=TERRA, tc="white", bold=True, fs=9, r=3, z=3)
    arrow(ax, (26.2, 38), (36, 31.5), GRY, style="-"); arrow(ax, (73.8, 38), (64, 31.5), GRY, style="-")
    arrow(ax, (26.2, 10), (36, 17), GRY, style="-"); arrow(ax, (73.8, 10), (64, 17), GRY, style="-")
    label(ax, 50, 42, "Apoios: retorno • glossário • comparação", 8.6, TEAL, True)
    label(ax, 50, 6, "Cuidados: autoria • fontes • dados", 8.6, TERRA, True)
    save(fig, "f19_sintese")

if __name__ == "__main__":
    f19_mapa(); f19_espectro(); f19_processo(); f19_sintese()
