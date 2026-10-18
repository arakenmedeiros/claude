from fe import *

def f20_mapa():
    mapa_conceitos("f20_mapa", "Matemática e\nciências com IA", [
        ("Como o sistema erra", ["texto plausível", "sensível ao enunciado"]),
        ("Ferramentas certas", ["calculadora, planilha", "simulações"]),
        ("Raciocínio do estudante", ["tentar primeiro", "dicas, não respostas"]),
        ("Verificação", ["estimar", "segunda via", "registrar"]),
        ("Modelagem e investigação", ["dados reais", "crítica do modelo"]),
        ("Segurança e dados", ["protocolos humanos", "sem dados pessoais"]),
    ], w=8.6, h=6.0, cor=BLUE)

def f20_papeis():
    comparar_cartoes("f20_papeis", [
        ("Ferramentas determinísticas", ["calculadora e planilha", "geometria dinâmica", "simulações com modelo explícito", "seguem regras exatas"], TEAL),
        ("Modelos de linguagem", ["geram texto plausível", "propõem exemplos e perguntas", "podem errar com segurança", "exigem verificação"], TERRA),
        ("Professor e estudantes", ["decidem o que perguntar", "interpretam e criticam", "validam com dados", "respondem pelo resultado"], PLUM),
    ], w=8.6, h=4.2, wrap_at=22)

def f20_verificar():
    fluxo("f20_verificar", [
        ("Estimar", "ordem de grandeza e sinal esperados"),
        ("Unidades", "as unidades fecham? faz sentido?"),
        ("Casos simples", "números fáceis e valores extremos"),
        ("Segunda via", "substituir de volta ou outro método"),
        ("Registrar", "escrever o que foi checado e como"),
    ], w=8.6, cor="multi", wrap_at=15, por_linha=5, h=2.7)

def f20_sintese():
    ciclo("f20_sintese", [("Explorar", "perguntas e exemplos"), ("Modelar", "dados reais"),
                          ("Verificar", "rotina de 5 passos"), ("Explicar", "com as próprias palavras")],
          centro="Raciocínio do\nestudante", w=7.2, h=5.0, cor=BLUE)

if __name__ == "__main__":
    f20_mapa(); f20_papeis(); f20_verificar(); f20_sintese()
