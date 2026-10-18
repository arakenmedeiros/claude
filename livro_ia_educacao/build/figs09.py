from fe import *

def f09_mapa():
    mapa_conceitos("f09_mapa", "Inclusão,\nacessibilidade e\npersonalização", [
        ("Marcos de referência", ["LBI, política, AEE", "DUA é recomendação"]),
        ("Desenho Universal (DUA)", ["engajamento", "representação, expressão"]),
        ("Apoios da IA", ["descrições, legendas", "glossários, variações"]),
        ("Validação por pessoas", ["estudantes e famílias", "equipes de apoio"]),
        ("Personalização", ["adaptação útil", "rótulos e promessas"]),
        ("Riscos", ["dados sensíveis", "infantilização"]),
    ], w=8.6, h=6.0, cor=TEAL)

def f09_udl():
    comparar_cartoes("f09_udl", [
        ("Engajamento: o porquê", ["Despertar interesse", "Sustentar esforço e persistência", "Apoiar a autorregulação"], TERRA),
        ("Representação: o quê", ["Texto, imagem, áudio, vídeo", "Glossários e esquemas", "Diferentes línguas e formatos"], TEAL),
        ("Ação e expressão: o como", ["Escrita, fala, desenho", "Produção multimídia", "Apoios e tempo adequados"], PLUM)],
        w=8.6, h=3.4, wrap_at=26)

def f09_personaliza():
    semaforo("f09_personaliza", [
        ("v", "Adaptação útil", "Ajusta apoio e tarefa com base em evidências de aprendizagem, com decisão do professor e possibilidade de recusa pelo estudante."),
        ("a", "Classificação inadequada", "Rótulos fixos (estilos, perfis, níveis) e trilhas que limitam oportunidades e expectativas."),
        ("r", "Alegação sem evidência", "\"A IA personaliza para cada aluno e melhora a nota\", sem estudos independentes, populações, resultados e prazos.")], w=8.6, wrap_at=95)

def f09_valida():
    fluxo("f09_valida", [("Objetivo", "e o que não pode ser perdido"), ("Gerar versões", "sem dados pessoais"), ("Verificar", "fidelidade e erros"), ("Validar", "com estudantes e equipe"), ("Registrar", "e oferecer como opções")], w=8.6, cor="multi", wrap_at=15, h=2.9)

def f09_sintese():
    fig, ax = canvas(8.2, 3.6, (0, 100), (0, 42))
    centro = [("Ouvir", "estudantes, famílias\ne equipes", TERRA), ("Proteger", "dados sensíveis\ne dignidade", TEAL), ("Validar", "fidelidade e\nadequação", PLUM), ("Manter alternativas", "sem IA e com\npoucos recursos", BLUE)]
    for i, (t, d, c) in enumerate(centro):
        x = 3 + i * 24.5
        rbox(ax, x, 6, 22, 30, "", fc=WHITE, ec=c, lw=2.2, r=3, z=2)
        rbox(ax, x, 27.5, 22, 8.5, t, fc=c, tc="white", bold=True, fs=8.6, r=3, z=3, wrap_at=16)
        ax.text(x + 11, 16, d, ha="center", va="center", fontsize=7.8, zorder=5, linespacing=1.2)
    label(ax, 50, 2.4, "Adaptar com responsabilidade: a IA apoia, as pessoas decidem.", 8.4, GRY, it=True)
    save(fig, "f09_sintese")

if __name__ == "__main__":
    f09_mapa(); f09_udl(); f09_personaliza(); f09_valida(); f09_sintese()
