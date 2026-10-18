# Registro de produção e pendências

## O que existe (arquivos)
- `Backpropagation_e_Redes_Neurais.docx`: livro completo (Partes 1 e 2, Caps. 1–8, projeto integrador, Apêndices A–K, referências), com sumário em campo TOC e números de página conferidos por duas passagens de diagramação (LibreOffice).
- `slides/`: 8 arquivos PPTX (um por capítulo, com notas do apresentador). Validados com `validate.py`; só o Cap. 8 foi inspecionado visualmente.
- `codigo/`: pacote `rnbook`, scripts por capítulo, `testes_rnbook.py`, `requirements.txt`. `notebooks/`: 8 `.ipynb`.
- `cap/`: fontes (Markdown, questões, exercícios práticos); `build/`: construtores do DOCX/PPTX/notebooks.

## Limitações declaradas
- Livro de referência (Oliveira e Melo Júnior, ISBN 9786558423515): sem acesso ao conteúdo; nada é atribuído a ele.
- Artigos originais (Fahlman 1988, Jacobs 1988, Riedmiller e Braun 1993, Igel e Hüsken 2000/2003, Minai e Williams 1990, Hagan e Menhaj 1994) NÃO foram lidos. As referências foram escritas de memória e NÃO foram conferidas em bases externas (a rede bloqueou); conferir antes de citar.
- UCI e Kaggle inacessíveis: experimentos usam bases do scikit-learn e dados sintéticos; as extensões com UCI/Kaggle estão sugeridas e NÃO foram executadas.
- O livro não trata mais de MATLAB: as correspondências são com o scikit-learn 1.9.1 e o SciPy (Apêndice I). Solvers e assinaturas foram conferidos no código instalado; o comportamento numérico não foi comparado algoritmo a algoritmo.
- Notebooks foram gerados dos scripts e NÃO foram executados como notebooks (sem saídas). Os números do texto vêm da execução dos `.py`.
- Resultados numéricos: NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1; poucas sementes; não são afirmações universais.
- Revisão final: não foi feita leitura integral página a página do DOCX renderizado nem inspeção visual de todos os PPTX; recomenda-se revisão humana.
- Hiperparâmetros de várias comparações foram ajustados com poucas sementes; ver ressalvas em cada capítulo.
