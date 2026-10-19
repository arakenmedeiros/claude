"""Capítulo 18 (Projeto 3): normas FICTÍCIAS da 'Prefeitura de Vila Aurora' com vigência, versões e perfis de acesso.

Todos os atos (números, datas, valores, prazos) são inventados. NÃO representam nenhum município real nem a legislação brasileira.
Formato: NORMAS = (id, categoria, título, texto); META[id] = dict(numero, inicio, fim, perfil, substitui). Datas ISO 'AAAA-MM-DD';
`fim` = None significa 'sem data de término'; o intervalo é [inicio, fim) (a norma deixa de valer NO dia `fim`).
CONSULTAS_NORMAS = (pergunta, data_de_referencia, perfil, relevantes, trecho, tipo); tipo in {simples, vigencia, restrito, sem_resposta}.
"""
NORMAS = [
 ("D201", "Expediente", "Portaria 12/2023 - Expediente das repartições", "Portaria 12/2023: o expediente das repartições municipais é das 8h às 14h, de segunda a sexta."),
 ("D202", "Expediente", "Portaria 45/2025 - Expediente das repartições", "Portaria 45/2025: o expediente das repartições municipais é das 8h às 16h, com atendimento ao público até as 15h. Revoga a Portaria 12/2023."),
 ("D203", "Diárias", "Portaria 18/2024 - Diárias de viagem", "Portaria 18/2024: a diária de viagem dentro do estado é de R$ 180,00 por dia."),
 ("D204", "Diárias", "Portaria 9/2026 - Diárias de viagem", "Portaria 9/2026: a diária de viagem dentro do estado é de R$ 210,00 por dia. Substitui a Portaria 18/2024."),
 ("D205", "Requerimentos", "Instrução Normativa 03/2024 - Requerimentos administrativos", "Instrução Normativa 03/2024: o prazo de resposta a requerimentos administrativos é de 30 dias, prorrogável por mais 30 dias mediante justificativa."),
 ("D206", "Pessoal", "Portaria 30/2025 - Teletrabalho", "Portaria 30/2025: servidores em funções elegíveis podem trabalhar em teletrabalho até 2 dias por semana, com autorização da chefia."),
 ("D207", "Pessoal", "Portaria 31/2025 - Férias de servidores", "Portaria 31/2025: o servidor pode parcelar as férias em até 3 períodos, nenhum inferior a 5 dias."),
 ("D208", "Transparência", "Instrução Normativa 05/2025 - Acesso à informação", "Instrução Normativa 05/2025: os pedidos de acesso à informação têm resposta em até 20 dias, prorrogáveis por mais 10 dias."),
 ("D209", "Corregedoria", "Portaria 50/2025 - Sindicância investigativa", "Portaria 50/2025: a sindicância investigativa deve ser concluída em 60 dias, e seu conteúdo é restrito à corregedoria."),
 ("D210", "Corregedoria", "Portaria 51/2025 - Processo disciplinar", "Portaria 51/2025: o prazo para conclusão do processo administrativo disciplinar é de 120 dias."),
 ("D211", "Tributos", "Decreto 7/2024 - Taxa de alvará", "Decreto 7/2024: a taxa de alvará de funcionamento é de R$ 320,00 e o alvará vale por 12 meses."),
 ("D212", "Tributos", "Decreto 14/2026 - Taxa de alvará", "Decreto 14/2026: a taxa de alvará de funcionamento é de R$ 350,00 e o alvará vale por 24 meses. Revoga o Decreto 7/2024."),
 ("D213", "Frota", "Portaria 22/2025 - Veículos oficiais", "Portaria 22/2025: o uso de veículo oficial exige autorização prévia e registro da quilometragem."),
 ("D214", "Compras", "Instrução Normativa 07/2025 - Compras diretas", "Instrução Normativa 07/2025: compras diretas de pequeno valor são admitidas até R$ 12.000,00 por contratação, com pesquisa de preços de no mínimo 3 fornecedores."),
 ("D215", "Transparência", "Portaria 40/2025 - Ouvidoria municipal", "Portaria 40/2025: a ouvidoria municipal responde às manifestações em até 15 dias."),
 ("D216", "Tecnologia", "Portaria 60/2026 - Processo eletrônico", "Portaria 60/2026: os processos administrativos passam a tramitar exclusivamente no sistema Aurora Docs a partir de 1º de julho de 2026."),
 ("D217", "Privacidade", "Portaria 61/2026 - Encarregado de dados", "Portaria 61/2026: o encarregado pelo tratamento de dados pessoais do município atende pelo e-mail privacidade@aurora.example."),
]

_P, _S, _J = "publico", "servidor", "juridico"
META = {
 "D201": dict(numero="12/2023", inicio="2023-03-01", fim="2025-02-01", perfil=_P),
 "D202": dict(numero="45/2025", inicio="2025-02-01", fim=None, perfil=_P, substitui="D201"),
 "D203": dict(numero="18/2024", inicio="2024-01-01", fim="2026-01-01", perfil=_S),
 "D204": dict(numero="9/2026", inicio="2026-01-01", fim=None, perfil=_S, substitui="D203"),
 "D205": dict(numero="IN 03/2024", inicio="2024-06-01", fim=None, perfil=_P),
 "D206": dict(numero="30/2025", inicio="2025-05-01", fim=None, perfil=_S),
 "D207": dict(numero="31/2025", inicio="2025-05-01", fim=None, perfil=_S),
 "D208": dict(numero="IN 05/2025", inicio="2025-03-01", fim=None, perfil=_P),
 "D209": dict(numero="50/2025", inicio="2025-08-01", fim=None, perfil=_J),
 "D210": dict(numero="51/2025", inicio="2025-08-01", fim=None, perfil=_J),
 "D211": dict(numero="7/2024", inicio="2024-01-01", fim="2026-03-01", perfil=_P),
 "D212": dict(numero="14/2026", inicio="2026-03-01", fim=None, perfil=_P, substitui="D211"),
 "D213": dict(numero="22/2025", inicio="2025-04-01", fim=None, perfil=_S),
 "D214": dict(numero="IN 07/2025", inicio="2025-04-01", fim=None, perfil=_S),
 "D215": dict(numero="40/2025", inicio="2025-06-01", fim=None, perfil=_P),
 "D216": dict(numero="60/2026", inicio="2026-07-01", fim=None, perfil=_P),
 "D217": dict(numero="61/2026", inicio="2026-04-01", fim=None, perfil=_P),
}

# hierarquia de perfis: quem pode ler o quê
NIVEL = {"publico": 0, "servidor": 1, "juridico": 2}

# (pergunta, data de referência, perfil do usuário, relevantes, trecho esperado, tipo)
CONSULTAS_NORMAS = [
 ("Qual o horário de expediente das repartições municipais?", "2024-06-01", "publico", ["D201"], "8h às 14h", "vigencia"),
 ("Qual o horário de expediente das repartições municipais?", "2026-05-01", "publico", ["D202"], "8h às 16h", "vigencia"),
 ("Qual o horário de expediente das repartições municipais?", "2025-02-01", "publico", ["D202"], "8h às 16h", "vigencia"),
 ("Qual a taxa de alvará de funcionamento?", "2025-06-01", "publico", ["D211"], "R$ 320,00", "vigencia"),
 ("Qual a taxa de alvará de funcionamento?", "2026-06-01", "publico", ["D212"], "R$ 350,00", "vigencia"),
 ("Por quantos meses vale o alvará de funcionamento?", "2026-10-01", "publico", ["D212"], "24 meses", "vigencia"),
 ("Qual o valor da diária de viagem dentro do estado?", "2025-06-01", "servidor", ["D203"], "R$ 180,00", "vigencia"),
 ("Qual o valor da diária de viagem dentro do estado?", "2026-06-01", "servidor", ["D204"], "R$ 210,00", "vigencia"),
 ("Em que sistema tramitam os processos administrativos?", "2026-10-01", "publico", ["D216"], "Aurora Docs", "vigencia"),
 ("Em que sistema tramitam os processos administrativos?", "2026-05-01", "publico", [], "", "vigencia"),
 ("Qual o prazo de resposta a requerimentos administrativos?", "2026-10-01", "publico", ["D205"], "30 dias", "simples"),
 ("Em quantos dias respondem aos pedidos de acesso à informação?", "2026-10-01", "publico", ["D208"], "20 dias", "simples"),
 ("Em quantos dias a ouvidoria municipal responde?", "2026-10-01", "publico", ["D215"], "15 dias", "simples"),
 ("Qual o e-mail do encarregado de dados pessoais?", "2026-10-01", "publico", ["D217"], "privacidade@aurora.example", "simples"),
 ("Quantos dias de teletrabalho por semana são permitidos?", "2026-10-01", "servidor", ["D206"], "2 dias por semana", "simples"),
 ("Em quantos períodos posso parcelar as férias?", "2026-10-01", "servidor", ["D207"], "3 períodos", "simples"),
 ("Até que valor posso fazer compra direta?", "2026-10-01", "servidor", ["D214"], "R$ 12.000,00", "simples"),
 ("O uso de veículo oficial exige autorização?", "2026-10-01", "servidor", ["D213"], "autorização prévia", "simples"),
 ("Qual o prazo da sindicância investigativa?", "2026-10-01", "juridico", ["D209"], "60 dias", "restrito"),
 ("Qual o prazo do processo administrativo disciplinar?", "2026-10-01", "juridico", ["D210"], "120 dias", "restrito"),
 ("Qual o prazo da sindicância investigativa?", "2026-10-01", "publico", [], "", "restrito"),
 ("Qual o prazo da sindicância investigativa?", "2026-10-01", "servidor", [], "", "restrito"),
 ("Qual o prazo do processo administrativo disciplinar?", "2026-10-01", "publico", [], "", "restrito"),
 ("Quantos dias de teletrabalho por semana são permitidos?", "2026-10-01", "publico", [], "", "restrito"),
 ("Qual o valor da diária de viagem dentro do estado?", "2026-10-01", "publico", [], "", "restrito"),
 ("Até que valor posso fazer compra direta?", "2026-10-01", "publico", [], "", "restrito"),
 ("Qual o salário do prefeito?", "2026-10-01", "publico", [], "", "sem_resposta"),
 ("Quantos servidores a prefeitura tem?", "2026-10-01", "servidor", [], "", "sem_resposta"),
 ("Quando será a próxima licitação de obras?", "2026-10-01", "publico", [], "", "sem_resposta"),
 ("Qual o horário de funcionamento do cemitério municipal?", "2026-10-01", "publico", [], "", "sem_resposta"),
]
