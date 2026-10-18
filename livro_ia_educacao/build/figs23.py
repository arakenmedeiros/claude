from fe import *

def f23_mapa():
    mapa_conceitos("f23_mapa", "Acessibilidade, trabalho\ne formação docente", [
        ("Marcos e princípios", ["direitos e legislação", "desenho universal"]),
        ("Materiais acessíveis", ["estrutura e linguagem", "alternativas perceptíveis"]),
        ("Proteção de dados", ["dados sensíveis", "sem laudos na IA"]),
        ("Organização do trabalho", ["triagem de tarefas", "banco verificado"]),
        ("Desenvolvimento profissional", ["ciclos entre pares", "formação sustentada"]),
        ("Riscos e evidências", ["adaptação errada", "evidências limitadas"]),
    ], w=8.6, h=6.0, cor=PLUM)

def f23_camadas():
    camadas("f23_camadas", [
        ("1. Estrutura do material", "títulos, listas, ordem lógica"),
        ("2. Alternativas perceptíveis", "texto alternativo, legendas, contraste"),
        ("3. Linguagem clara", "frases diretas, termos explicados"),
        ("4. Apoios e tempo", "formatos, prazos, instruções por etapas"),
        ("5. Participação", "o estudante decide o que o ajuda"),
    ], w=7.6, h=5.2)

def f23_fluxo():
    fluxo("f23_fluxo", [
        ("Triagem", "usar, adaptar ou dispensar a IA"),
        ("Rascunho", "pedido claro, sem dados pessoais"),
        ("Verificação", "conferir com o original"),
        ("Adaptação", "ajustar à turma e testar"),
        ("Registro", "guardar no banco pessoal"),
    ], w=8.6, h=2.6, cor="multi", wrap_at=14)

def f23_sintese():
    fig, ax = canvas(8.6, 4.4, (0, 100), (0, 51))
    xs = [2, 35, 68]
    ts = [("Acessibilidade", "materiais verificados\ndesenho universal", TEAL), ("Trabalho", "triagem, banco pessoal\nlimites de tempo", TERRA), ("Formação", "ciclos entre pares\nregistro e reflexão", PLUM)]
    for x, (t, d, c) in zip(xs, ts):
        rbox(ax, x, 18, 30, 26, "", fc=WHITE, ec=c, lw=2.2, r=3, z=2)
        rbox(ax, x, 36, 30, 8, t, fc=c, tc="white", bold=True, fs=9.5, r=3, z=3)
        label(ax, x + 15, 26, d, 8.2)
    rbox(ax, 2, 2, 96, 9, "IA: rascunhos sob verificação do professor", fc=SAND, ec=BLUE, tc=BLUE, fs=9, bold=True, r=3, z=3)
    for x in (17, 50, 83):
        arrow(ax, (x, 11.2), (x, 17.8), BLUE, ls="--", lw=1.4)
    save(fig, "f23_sintese")

if __name__ == "__main__":
    f23_mapa(); f23_camadas(); f23_fluxo(); f23_sintese()
