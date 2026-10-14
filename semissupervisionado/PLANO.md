# Plano e estado do livro "Aprendizado Semissupervisionado"

Autor: Araken de Medeiros Santos. Título: *Aprendizado Semissupervisionado: do zero à prática com Python e scikit-learn*. Público misto, sem pré-requisitos. ABNT. Docx final: `Aprendizado_Semissupervisionado.docx`.

## Capítulos (arquivos em cap/)
| Cap | Título | Estado |
|---|---|---|
| 1-3 | Parte I Fundamentos | PRONTO |
| 4 | Self-training | PRONTO |
| 5 | Co-training e multivisão | PRONTO |
| 6 | Modelos generativos e EM | PRONTO |
| 7 | Baixa densidade: S3VM, TSVM, entropia | PRONTO |
| 8 | Grafos: Label Propagation/Spreading | PRONTO |
| 9 | Redes neurais essenciais | |
| 10 | Regularização por consistência | |
| 11 | Pseudo-rótulos modernos (MixMatch, FixMatch) | |
| 12 | Auto-supervisionado e contrastivo | |
| 13 | Generativos profundos (VAE, GAN) | |
| 14 | Texto | |
| 15 | Imagens | |
| 16 | Séries temporais e tabulares | |
| 17 | Avaliação, armadilhas, escolha, ética | |
| 18 | Projeto integrador final (`@final`) | |

Partes: `@part:` só no 1º capítulo de cada parte: ch04 "Parte II — Métodos Clássicos", ch09 "Parte III — Métodos Modernos com Aprendizado Profundo", ch14 "Parte IV — Prática e Aplicações", ch18 "Parte V — Fechamento".

## Arquivos por capítulo
- `cap/chNN.md`: texto. Meta: `@part:`, `@banner: N`, `# título`, `@obj:` (várias). Seções `## N.M`, `### N.M.K`. Quadros `:::dica|atencao|cotidiano|saibamais|exemplo|memorize Título` ... `:::`. Figura: `![id](Figura N.M — legenda.)`. Fórmula: `$$ ... $$` (subscrito `_x`/`_(...)`, sobrescrito `^2`). Tabela `|`. Citação `> `.
- **Código**: `@@code nome` no md insere `cap/code/nome.py` + saída real `nome.out`. Rodar `python3 build/run_code.py chNN` (cwd=raiz do projeto; cache por hash; `--force`). Nunca escrever saída à mão. Scripts autocontidos; dados baixados para `dados/` (gitignored) via função `baixar(url, destino)`.
- `cap/exNN.txt`: 30 `MC:` (4 alternativas a)-d), linha `R: x | justificativa`) + 20 `DISC:` (linha `R: resposta`). Validar: `python3 build/chk.py`.
- `cap/pr/prNN.txt`: `@@EX n | título | nível`, enunciado, `@@SOL` com bloco ```python, `@@NOTE`; 5 exercícios + `@@PROJ título | CapNN` (+ `@@RUBRICA`) + `@@ALEM`. Rodar `python3 build/runpr.py cap/pr/prNN.txt` (grava build/pr_out.json). Notas devem refletir as saídas reais.
- Figuras: `build/figs_*.py` (usam `fc.py`; salvam em `figs/`; ids `fN_nome`); banners em `build/figs_cover.py` (`banner(n)`, PART colors até 17; ajustar para 18).
- `cap/glossario.txt`, `cap/referencias.txt` (ABNT; sem ordinais de NeurIPS, usar volume), `cap/guia.md` (Apêndice C), `cap/apresentacao.md`.
- Build: `bash build/make.sh` (2 passes p/ paginação; valida estabilidade). Validar docx: skill docx `scripts/office/validate.py`.

## Dados (acesso: raw.githubusercontent.com funciona; UCI/Kaggle/OpenML bloqueados)
Usar espelhos públicos de bases originais UCI/Kaggle e dizer isso com franqueza. Sklearn embutidos: Iris, Wine, Breast Cancer, Digits. Espelhos testados (HTTP 200): MNIST `mnist.pkl.gz` (mnielsen/neural-networks-and-deep-learning), Fashion-MNIST idx.gz (zalandoresearch/fashion-mnist `data/fashion/`), SMS Spam `justmarkham/DAT8/master/data/sms.tsv`, jbrownlee/Datasets: `pima-indians-diabetes.data.csv`, `sonar.csv`, `ionosphere.csv`, `adult-all.csv`, `glass.csv`, `daily-min-temperatures.csv`, `airline-passengers.csv`, `monthly-sunspots.csv`.

## Regras de honestidade (manter)
- Todo número do texto vem de execução real. Reportar resultados negativos. Referências de memória, avisar. Sem inventar.
- Sem pull request. Commit/push na branch `claude/vigilant-fermi-gbu78w` ao fim de cada parte.
