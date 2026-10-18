from fe import *

def f15_mapa():
    mapa_conceitos("f15_mapa", "IA na EPT", [
        ("Características", ["competências, saber fazer", "normas e segurança"]),
        ("IA como ferramenta", ["usar, avaliar, questionar", "no trabalho"]),
        ("IA como apoio", ["estudar e simular", "documentar"]),
        ("Verificação técnica", ["norma e manual", "profissional habilitado"]),
        ("Avaliação", ["demonstração prática", "requisitos de segurança"]),
        ("Riscos", ["segurança e responsabilidade", "confidencialidade"]),
    ], w=8.6, h=6.0, cor=GOLD)

def f15_lugares():
    comparar_cartoes("f15_lugares", [
        ("IA como objeto e ferramenta de trabalho", ["Formar para usar, avaliar e questionar", "Diagnóstico, previsão de falhas, atendimento, código, design", "Limites e responsabilidades"], TERRA),
        ("IA como apoio ao ensino", ["Estudar e simular situações", "Preparar materiais e documentar práticas", "Sempre com verificação"], TEAL)],
        w=8.6, h=3.2, wrap_at=40)

def f15_sintese():
    fluxo("f15_sintese", [("Resposta da IA", "procedimento ou explicação"), ("Conferência", "norma e manual"), ("Validação", "profissional habilitado"), ("Prática supervisionada", "só depois")], w=8.6, cor="multi", wrap_at=16, h=2.9)

if __name__ == "__main__":
    f15_mapa(); f15_lugares(); f15_sintese()
