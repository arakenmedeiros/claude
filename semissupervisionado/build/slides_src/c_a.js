module.exports = {
1: {
  gancho: "Imagine uma biblioteca com 100.000 livros e um único bibliotecário. Ele só consegue etiquetar 100 por semana. O resto fica sem etiqueta. Será que as etiquetas dos 100 livros ajudam a adivinhar as dos outros 99.900?",
  ganchoPergunta: "Essa é a pergunta do livro inteiro.",
  slides: [
    { t: "O que é 'aprender com dados'?", bullets: ["Cada exemplo tem atributos X (as características)", "E, às vezes, um rótulo y (a resposta certa)", "Classificação: y é uma categoria (spam / não spam)", "Regressão: y é um número (preço)"], analogia: "X é a ficha do aluno; y é a nota. Treinar é aprender a prever a nota pela ficha.", nota: "Defina atributo, rótulo, classificação e regressão com exemplos do cotidiano. Insista que o rótulo é a resposta correta, conhecida só para alguns exemplos." },
    { t: "Treino e teste: a honestidade do aprendizado", bullets: ["Treino: onde o modelo aprende", "Teste: provas que ele nunca viu", "Nota boa no treino não garante nada", "O teste fica trancado até o fim"], analogia: "Estudar com o gabarito e fazer a prova com o mesmo gabarito não mede aprendizado.", nota: "Esta é a regra de ouro do livro: nunca usar o teste para decidir nada." },
    { t: "Os paradigmas de aprendizado", fig: "f1_tipos", bullets: ["Supervisionado: todos rotulados", "Não supervisionado: nenhum rótulo", "Semissupervisionado: alguns rótulos", "Auto-supervisionado: rótulo inventado dos próprios dados"], nota: "Mostre a figura como um espectro de quantidade de rótulos. O semissupervisionado vive no meio." },
    { t: "O problema: rótulos são caros", fig: "f1_iceberg", bullets: ["Radiologista laudando tomografias", "Horas de áudio para transcrever", "Dados sem rótulo se acumulam de graça"], analogia: "Rótulos são a ponta do iceberg: poucos, visíveis e caros.", nota: "Dê exemplos de custo de rotulagem: tempo de especialistas, erros humanos, privacidade." },
    { t: "Definindo o problema com precisão", bullets: ["Temos X para todos os n exemplos", "Temos y só para l deles (l ≪ n)", "Indutivo: modelo para qualquer exemplo novo", "Transdutivo: rotular o conjunto sem rótulo que já existe"], analogia: "Indutivo é aprender a regra; transdutivo é corrigir só as provas que estão na mesa.", nota: "Diferença indutivo × transdutivo reaparece no Cap. 3 e no Label Spreading (Cap. 8)." },
    { t: "O que significa 'ajudar'?", bullets: ["Superar o modelo só com os rotulados", "No mesmo conjunto de teste", "Com a mesma família de modelo", "Sem referência, não há ganho a medir"], tipo: "texto", analogia: "Dizer que um remédio funciona exige comparar com quem não tomou.", nota: "Introduza o conceito de modelo de referência (baseline), que será insistentemente cobrado ao longo do livro." },
    { t: "Seu primeiro experimento: as duas luas", fig: "f1_luas", bullets: ["Só 4 pontos rotulados (as estrelas)", "Os sem rótulo mostram a forma das luas", "Um método semissupervisionado segue a curva"], nota: "Mostre que, com poucos rótulos, a fronteira supervisionada é uma reta que corta as luas; os dados sem rótulo revelam o formato." },
    { t: "Onde isso é usado?", fig: "f1_aplicacoes", tipo: "figura", bullets: ["Saúde", "Texto", "Imagens", "Voz", "Fraudes"], nota: "Peça à turma exemplos do seu próprio setor." },
    { t: "Um projeto semissupervisionado, passo a passo", fig: "f1_projeto", tipo: "figura", analogia: "Como uma receita: separe os ingredientes (teste) antes de cozinhar.", nota: "Etapas: problema, dados, teste separado, referência, método, avaliação, relato." },
    { tipo: "frase", texto: "Dados sem rótulo não são mágica.", sub: "Ajudam quando a forma dos dados tem relação com as classes e podem atrapalhar quando não tem.", nota: "Frase de impacto que atravessa o livro." },
  ],
  frase: "Antes de acreditar em um ganho, pergunte: “em comparação com o quê?”",
  proximo: "Próximo capítulo: a caixa de ferramentas — Python, probabilidade e avaliação.",
},
};
