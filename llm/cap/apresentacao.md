# Apresentação
## Para quem é este livro

Este livro foi escrito para um **público misto**: estudantes de graduação e de pós-graduação, profissionais de tecnologia e pessoas curiosas que querem entender como funcionam os grandes modelos de linguagem (LLMs) e como construir aplicações com eles.

Você **não precisa** de conhecimentos prévios de aprendizado de máquina. Os conceitos de matemática e de programação de que o livro depende são introduzidos e revisados no Capítulo 2, e cada termo técnico essencial é explicado no momento em que aparece. Os capítulos começam no nível básico, passam pelo intermediário e chegam a temas avançados, como ajuste fino eficiente, aplicações multimodais, uso de ferramentas e avaliação de sistemas.

## Como o livro está organizado

O livro tem **cinco partes** e **dezoito capítulos**.

- **Parte I — Fundamentos:** o que são os LLMs; matemática e computação essenciais; texto, tokens e embeddings; atenção e Transformers.
- **Parte II — Treinamento, adaptação e geração:** como os modelos aprendem; geração de texto e inferência; ajuste por instruções e preferências.
- **Parte III — Prompts e aplicações:** engenharia de prompts; aplicações em Python; embeddings e busca semântica; RAG.
- **Parte IV — Adaptação e sistemas avançados:** ajuste fino eficiente; IA multimodal; ferramentas, fluxos e agentes; eficiência e implantação.
- **Parte V — Avaliação e uso responsável:** avaliação de modelos e aplicações; segurança, ética e governança; projetos integradores.

Cada capítulo segue o mesmo roteiro: situação inicial; objetivos de aprendizagem; pré-requisitos; mapa conceitual; desenvolvimento; exemplos resolvidos; atividade prática; erros frequentes; limitações e cuidados; síntese; **30 questões de múltipla escolha**; **20 questões discursivas**; exercícios práticos em Python; um projeto; e as referências do capítulo. O **Capítulo 18** reúne seis projetos integradores.

## Ferramentas pedagógicas

Ao longo do texto, você encontrará quadros com funções específicas.

- **Intuição:** a ideia por trás do conceito, em linguagem cotidiana.
- **Passo a passo:** cálculos e procedimentos detalhados.
- **Na prática:** conselhos de quem constrói sistemas.
- **Cuidado com esta interpretação:** erros de leitura comuns.
- **O que vale a pena lembrar:** o essencial para memorizar.

## Código, dados e reprodutibilidade

Todos os códigos do livro rodam em **CPU**, sem internet, com Python, NumPy e scikit-learn. Os dados usados são **sintéticos ou fictícios**, escritos para o livro e reunidos no pacote `llmdidatico`, que acompanha o material. Cada bloco de código vem identificado:

- **PYTHON · EXECUTADO:** foi realmente executado na elaboração do livro, e a **saída** mostrada logo abaixo é a saída obtida.
- **PYTHON · NÃO EXECUTADO NESTE AMBIENTE:** código ilustrativo para modelos e serviços que o ambiente de elaboração não permitia acessar (por exemplo, modelos de grande escala e APIs). Ele foi escrito com base na documentação oficial, mas **não foi testado**; confira as versões antes de usar.
- **PSEUDOCÓDIGO** e **EXEMPLO ILUSTRATIVO:** esboços para explicar ideias.

O Apêndice H traz o guia de instalação e as versões das bibliotecas usadas. Serviços externos mudam sem aviso; por isso, resultados obtidos com eles podem não se repetir.

## Atualização e rigor bibliográfico

A revisão bibliográfica e a verificação de informações voláteis foram feitas até **outubro de 2026**. Modelos, preços, limites e interfaces de serviços mudam rapidamente; este livro evita depender deles e, quando os cita, indica a necessidade de conferir a documentação oficial. O Apêndice L descreve o método, o que foi verificado e o que permanece pendente. Referências marcadas com † têm dados que ainda devem ser conferidos na fonte original.

## Como estudar com este livro

- **Estude em blocos curtos.** Sessões de 25 a 40 minutos, com pausas, ajudam a manter a atenção.
- **Pare e explique.** Após cada seção, explique a ideia em voz alta, como se falasse com um colega.
- **Rode os códigos.** Altere um parâmetro por vez e observe o que muda.
- **Verifique sempre.** Respostas fluentes não são garantia de correção. Habitue-se a conferir.
- **Responda antes de olhar o gabarito.** Tentar lembrar fixa mais do que reler.
