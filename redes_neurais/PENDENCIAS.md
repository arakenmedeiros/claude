# Registro de produção e pendências

## O que existe (arquivos)
- `Backpropagation_e_Redes_Neurais.docx`, `.pdf` e `.epub`: livro completo (três partes, Caps. 1–9, projeto integrador, Apêndices A–L, referências), com texto justificado. O DOCX e o PDF têm sumário com números de página conferidos por duas passagens de diagramação (LibreOffice); o EPUB tem sumário navegável.
- `slides/` e `slides_redes_neurais.zip`: 9 arquivos PPTX (um por capítulo), com notas do apresentador.
- `codigo/`: pacote `rnbook` (listado no Apêndice L), scripts por capítulo, `testes_rnbook.py`, `requirements.txt` e `requirements-opcional.txt`. `notebooks/`: 9 `.ipynb` (sem saídas).
- `cap/`: fontes (Markdown, questões, exercícios práticos); `build/`: construtores do DOCX/EPUB/PPTX/notebooks.

## Revisão a partir do parecer (esta versão)
Corrigidos: (a) erro de treino abaixo da variância do ruído não prova sobreajuste; (b) `loss_` restrito à regressão com perda quadrática (classificador: perda logarítmica; Poisson: metade da deviance); (c) gradientes que desaparecem dependem também dos pesos; (d) direção mais lenta do gradiente descendente depende de $\eta$; (e) troca de sinal no RPROP tratada como heurística; (f) tratamento de `LinAlgError` no LM impresso (e salvaguarda de solução não finita no `rnbook`); (g) parada antecipada do scikit-learn (pontuação de validação; só `sgd`/`adam`); (h) posto numérico; LM usa $J^TJ$, não a Hessiana completa; "aproximação numérica da Hessiana completa"; (i) descrição do Adam; (j) tabelas dentro de blocos coloridos e remissões de equações (agora numeradas e por rótulo); (k) marcas do processo de elaboração removidas; (l) código do `rnbook` incorporado (Apêndice L). Acrescentado o Capítulo 9 (otimizadores modernos, grafos gerais e autodiferenciação, perdas estáveis, multirrótulo, produtos Hessiana–vetor, diagnóstico) com 30 questões de múltipla escolha e 20 discursivas (totais: 270 e 180).

## Limitações declaradas
- Artigos originais (Fahlman 1988, Jacobs 1988, Riedmiller e Braun 1993, Igel e Hüsken 2000/2003, Minai e Williams 1990, Hagan e Menhaj 1994) NÃO foram lidos na íntegra. As referências foram escritas a partir do conhecimento do autor e NÃO foram conferidas por mim em bases externas (rede bloqueada); um parecer externo confirmou a existência de várias delas. Confira antes de citar páginas.
- As novas referências do Capítulo 9 (Kingma e Ba; Loshchilov e Hutter; Duchi et al.; Tieleman e Hinton; Nesterov; Polyak; Srivastava et al.; Ioffe e Szegedy; Ba et al.; He et al.; Pearlmutter; Martens; Sutskever et al.; Paszke et al.; Bradbury et al.) também NÃO foram conferidas em bases externas.
- UCI e Kaggle inacessíveis: experimentos usam bases do scikit-learn e dados sintéticos; as extensões com UCI/Kaggle estão sugeridas e NÃO foram executadas.
- PyTorch NÃO foi executado; a verificação com diferenciação automática de biblioteca usou o JAX 0.12.1 (opcional).
- Notebooks gerados dos scripts e NÃO executados como notebooks (sem saídas); os números do texto vêm da execução dos `.py`.
- Resultados numéricos: NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1; poucas sementes; não são afirmações universais. Várias melhores configurações ficaram na borda das grades (indicado no texto).
- A paginação do sumário foi conferida com o LibreOffice; outros editores (por exemplo, o Word) podem paginar diferente: atualize o sumário (campo) ao abrir.
- Não houve leitura integral página a página do DOCX/PDF renderizado nem inspeção visual de todos os slides; recomenda-se revisão humana.
