from fe import *

def f26_mapa():
    mapa_conceitos("f26_mapa", "Verificar, proteger\ne registrar", [
        ("Tipos de erro", ["fabricação", "desatualização"]),
        ("Como verificar", ["fonte primária", "leitura lateral"]),
        ("Proporcionalidade", ["risco baixo a alto", "quem confere"]),
        ("Dados pessoais", ["minimizar", "anonimizar"]),
        ("Direitos e normas", ["LGPD", "autoria e licenças"]),
        ("Registro", ["ficha de uso", "declaração"]),
    ], w=8.6, h=6.0, cor=TEAL)

def f26_fluxo():
    fluxo("f26_fluxo", [("Separar", "listar afirmações verificáveis"), ("Localizar", "buscar fonte primária"),
                        ("Ler o trecho", "o texto sustenta a frase?"), ("Registrar", "fonte, data e decisão"),
                        ("Decidir", "usar, corrigir ou descartar")], w=8.6, h=2.6, cor="multi", wrap_at=16, por_linha=5)

def f26_dados():
    semaforo("f26_dados", [
        ("v", "Pode, em geral", "Planos de aula, textos próprios, situações fictícias, conteúdo curricular público."),
        ("a", "Só com cuidado e base adequada", "Produções de estudantes sem identificação, dados agregados da turma, texto de terceiros com licença."),
        ("r", "Não inserir em sistemas externos", "Nomes, laudos, notas individuais, imagens e voz de crianças, dados de participantes de pesquisa, documentos sigilosos.")],
        w=8.6, wrap_at=95)

def f26_sintese():
    fig, ax = canvas(8.6, 4.0, (0, 100), (0, 46))
    for x, t, c in [(3, "Verificar\n(fontes e conferência)", TERRA), (35, "Proteger\n(dados e direitos)", TEAL), (67, "Registrar\n(uso e decisões)", PLUM)]:
        rbox(ax, x, 20, 30, 16, t, fc=c, tc="white", bold=True, fs=9, r=3, z=3)
    rbox(ax, 3, 3, 94, 11, "Responsabilidade docente: eu decido, eu confiro, eu respondo pelo que uso", fc=SAND, bold=True, fs=9, r=3, z=3)
    save(fig, "f26_sintese")

if __name__ == "__main__":
    f26_mapa(); f26_fluxo(); f26_dados(); f26_sintese()
