# Revisão do livro conforme o parecer técnico

Correções aplicadas (arquivos em `cap/`):

| Item do parecer | Correção |
|---|---|
| FedSGD (Cap. 3) | Condições da equivalência com o passo centralizado (participação total, mesma perda e ponto inicial, lote inteiro, ponderação por n_k, sem regularização/compressão/ruído). |
| DP e número de clientes (Cap. 8, 13, exercícios, slides) | O ε depende de z, das rodadas e de δ, e não do número de clientes; mais clientes tornam a privacidade mais barata em utilidade. |
| Escopo da DP (quadro de proteção, Cap. 8) | Garantia delimitada ao mecanismo, à unidade protegida, à vizinhança e ao observador; não cobre envenenamento nem dados antes do mecanismo. |
| Vizinhança, denominador (Cap. 8, `privacidade.py`) | Adição/remoção (sensibilidade S) × substituição (2S); denominador constante pública; `epsilon_rdp(..., vizinhanca=)`. |
| Mediana, média aparada (Cap. 9) | Ponto de ruptura ≠ "não se move"; contraexemplos `[0,1,2,3,4]`; a aparada remove extremos, e não maliciosos. |
| Krum (Cap. 9, `robustez.py`) | Validação de m > 2f + 2 e discussão das hipóteses; exercício ajustado (f ≤ 8 com 20 clientes). |
| Média simples × FedAvg ponderado (Cap. 9) | Aviso explícito sobre pesos iguais e sobre o objetivo otimizado. |
| Validação × teste (Cap. 9) | Conjunto de validação para decisões do servidor; teste final reservado. |
| Inversão de gradientes com ruído (Cap. 8) | Aviso: a piora do ataque não demonstra DP. |
| Projeto integrador (agora Cap. 13, `projeto.py`) | Corte reaplicado após a quantização; denominador fixo; demonstração (0; 0,6; 0,8) → norma 1,131. |
| Ditto (Cap. 6, `personalizacao.py`) | `ditto` intercalado (algoritmo original) e `ajuste_proximal` (posterior, simplificação inspirada no Ditto); experimento com ambos. |
| Pré-processamento (Caps. 2, 3, 6, 13) | Hipótese declarada; `padronizar_clientes` usa só o treino dos clientes quando há teste local. |
| Referências | Tan et al. (TNNLS, v. 34, n. 12, 2023); páginas de McMahan, Ditto, Blanchard e Yin; novas referências verificadas. |

Temas incluídos: novo **Capítulo 12** (assincronia/FedBuff, seleção de clientes e viés de participação, avaliação em clientes novos e partições naturais, FedBN, desaprendizado/FedEraser, pseudorrótulos, multirrótulo e hierarquia, deriva de conceito), com código (`cap/lib/avancado.py`), experimentos, 30 questões de múltipla escolha, 20 discursivas, 4 exercícios práticos e um projeto. O projeto integrador passou a ser o Capítulo 13.
