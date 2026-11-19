# Registro de revisões — versão 2

Revisão do livro *Aprendizado Não Supervisionado* conforme o parecer recebido sobre a versão v01 (349 páginas, 13 capítulos).

## Correções conceituais prioritárias

| Ponto do parecer | Onde | O que foi feito |
|---|---|---|
| A. Biplot: perpendicularidade não é independência | Cap. 8 (8.7.2), ex. 8 | Reescrito: ângulos próximos de 90° sugerem correlação linear próxima de zero, sem demonstrar independência (exemplo *Y* = *X*²); depende da construção do gráfico e da variância preservada. Gabarito ajustado. |
| B. Espectral: variantes misturadas | Cap. 6 (6.10.2 a 6.10.5), glossário, ex. 6 | Procedimento principal passou a ser a variante de Ng, Jordan e Weiss (L_sym, normalização das linhas, K-means). Novo quadro compara L = D − W (RatioCut), L_rw (corte normalizado, Shi e Malik) e L_sym. |
| C. NT-Xent: denominador | Cap. 12 (12.9.2), ex. 12 | Equação com índices explícitos (k ≠ i); texto e gabarito dizem que o denominador inclui o positivo e os negativos, excluindo só a âncora. O código já estava correto. |
| D. Hopkins alto não prova vários grupos | Cap. 7 (7.2), ex. 7, pr. 7 | Interpretação reescrita em relação ao modelo nulo uniforme; novo quadro de atenção; o código ganhou um caso de uma única nuvem gaussiana (H ≈ 0,84). Exercícios e saídas atualizados. |
| E. Autoencoder para anomalias sem protocolo | Cap. 12 (12.4.5), ex. 12, pr. 12 | Novo protocolo: treino/validação/teste, escalonador só no treino, limiar na validação, normais inéditos no teste, AUC, precisão e revocação; aviso de que anomalias podem ser bem reconstruídas. |

## Outras afirmações ajustadas

- Distância do cosseno não é métrica (Cap. 2).
- K-means como caso-limite do GMM, com as condições (Cap. 6).
- HDBSCAN: `min_samples`; `probabilities_` como força de pertinência (Cap. 5).
- Estabilidade por subamostragem sem reposição, e não bootstrap (Cap. 7).
- `nu` do One-Class SVM (Cap. 11).
- Analogia do VAE qualificada (Cap. 12).
- Variância não é o mesmo que importância, já na primeira explicação do PCA (Cap. 8).

## Temas incluídos

Novo **Capítulo 13 — Agrupamento no Mundo Real** (Parte VI): consenso de agrupamentos, agrupamento com restrições, dados mistos (Gower e K-prototypes), biclustering, detecção de comunidades (Louvain e Leiden), fluxos e deriva de conceito, modelagem de tópicos e validação fora da amostra. Traz 6 figuras, 30 questões de múltipla escolha, 20 discursivas, 5 exercícios práticos e 1 projeto. O projeto final passou a ser o Capítulo 14 (Parte VII).

## Ajustes editoriais

- Sumário regenerado pela passada dupla de paginação.
- Numeração dos passos do PCA corrigida (sublistas dentro de listas numeradas).
- Grafia padronizada: "autossupervisionado".
- Equações: suporte a subscritos e sobrescritos entre chaves e a fórmulas dentro do texto.
- Texto justificado (DOCX, PDF e EPUB), com hifenização automática no DOCX e no EPUB.
- Cadernos Jupyter por capítulo, `requirements.txt` e instruções de execução em `codigo/`.
- 36 novas referências e 16 novos termos no glossário; guia rápido ampliado.
- Todos os blocos de código e soluções foram reexecutados nas versões listadas na Apresentação.
