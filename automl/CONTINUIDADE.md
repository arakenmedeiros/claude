# Continuidade do projeto — "AutoML: Teoria e Prática"

Documento de passagem para retomar o trabalho em outra sessão (por exemplo, no Cowork).

## 1. Projeto
- Livro didático "AutoML: Teoria e Prática". Autor: Araken de Medeiros Santos.
- Público misto, sem pré-requisitos. Português do Brasil, referências em ABNT.
- 14 capítulos em 5 partes e apêndices A–E.
- Por capítulo: 30 questões de múltipla escolha e 20 discursivas, com gabarito no Apêndice A; exercícios práticos em Python/scikit-learn e um projeto. O Cap. 14 traz o projeto integrador.
- Persona pedida pelo usuário: professor e pesquisador sênior em IA, tom didático e acolhedor.

## 2. Estado atual
- Livro revisado conforme o parecer do usuário e entregue em `.docx`, `.pdf` (452 páginas) e `.epub`, com texto justificado.
- Branch: `claude/gifted-ritchie-i7q8s4`. Último commit: `f29c70b`. Sem pull request (só criar se o usuário pedir).
- Pasta de trabalho: `automl/`.

## 3. Estrutura da pasta
- `src/chNN.md`: fontes dos capítulos, com blocos de código marcados. `cap/chNN.md` é gerado a partir deles.
- `cap/`: exercícios (`exNN.txt`), práticos (`pr/prNN.txt`), `referencias.txt`, `glossario.txt`, `guia.md`, `apresentacao.md`, `VERIFICACAO_REFERENCIAS.md`.
- `build/`: `gen.py`, `runpr.py`, `chk.py`, `build.js`, `make.sh`, `make_epub.sh`, `figs*.py`, `extrair_codigo.py`, `epub/epub.css`.
- `codigo/`: um script por bloco, exercícios, trechos não executados e README.
- `resultados/`: JSONs de saída.
- `slides/`: um `.pptx` por capítulo (`build.js`, `specs/`, `out/`).
- `requirements.txt`: versões fixadas (numpy 2.5.3, pandas 3.0.5, scipy 1.18.1, matplotlib 3.11.2, scikit-learn 1.9.1, optuna 5.0.0, flaml 2.7.0, shap 0.54.1, lightgbm 4.7.0, xgboost 3.4.1, statsmodels 0.15.0, joblib 1.6.0, threadpoolctl 3.8.0).

## 4. Como reconstruir
1. Instalar as dependências de `requirements.txt` (mais Node com `docx` e LibreOffice para o PDF; pandoc 3.1.3 para o EPUB).
2. `python build/gen.py`: executa os blocos Python (cache em `.outcache`, marcadores `#@saida`, `#@nb`, `#@nbx`, `#@naoexec`) e gera `cap/chNN.md`.
3. `python build/runpr.py`: roda os práticos e grava `pr_out/NN.json`.
4. `python build/chk.py`: confere que cada capítulo tem 30 MC e 20 discursivas.
5. `bash build/make.sh`: gera DOCX e PDF (duas passagens para o sumário).
6. `bash build/make_epub.sh`: gera o EPUB (HTML próprio via `EMIT_HTML=1` em `build.js`, depois pandoc).
7. `python build/extrair_codigo.py`: atualiza `codigo/`.

## 5. Decisões e restrições combinadas
- Referências só com dados confirmados; DOI ou URL não confirmados são omitidos. Registro em `cap/VERIFICACAO_REFERENCIAS.md`.
- Não criar PR sem pedido. Push só na branch acima, com `git push -u origin`.
- Commits terminam com os trailers de coautoria e de sessão da plataforma.
- Não incluir identificador de modelo em artefatos do repositório.
- Texto justificado em todos os formatos.
- Trechos de Auto-sklearn, TPOT (série 0.x), H2O, AutoGluon, Auto-Keras, TabPFN e FastAPI não foram executados; o livro avisa isso.
- Experimentos de ASHA, NAS com ruído, restrições e RAG são simulações ou toy problems, rotulados no texto.

## 6. Principais mudanças da última revisão
- Cap. 13: três partes (treino/validação/teste), limiares por grupo escolhidos na validação, paridade demográfica vs. igualdade de oportunidade, SHAP em log-odds, nova seção de calibração, predição conforme e abstenção.
- Cap. 14: análise pareada (bootstrap pareado e McNemar) para campeão/desafiante, AutoML com restrições.
- Cap. 10: benchmark com recursos iguais e medidos, TabPFN, TPOT identificado como série 0.x.
- Cap. 11: tempo de construção e variação das sementes no NAS.
- Cap. 12: preenchimento causal na série temporal, multirrótulo, LLM/RAG.
- Cap. 6: BOHB corrigido e ASHA. Caps. 2 e 3: desbalanceamento e target encoding.
- Bibliografia completada; versões fixadas; códigos extraídos.

## 7. Pendências
- Slides dos Caps. 2, 3, 6 e 10–14 estão desatualizados em relação à revisão. Para refazer: editar `slides/specs/chNN.json` e rodar `NODE_PATH=slides/node_modules node build.js specs/chNN.json out/<arquivo>.pptx`.
- Conferir DOIs e URLs omitidos nos sites das editoras (sem acesso daqui).
- Validar o EPUB com `epubcheck` (só foram verificados XML bem formado e dois trechos renderizados).
- Conferir o `.docx`/`.pdf` página a página (só foi feita amostragem visual).
