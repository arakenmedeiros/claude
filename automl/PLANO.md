# AutoML: Teoria e Prática — plano do livro

Autor: Araken de Medeiros Santos · Público: misto, sem pré-requisitos · Idioma: pt-BR · Referências: ABNT · Saída: .docx

## Estrutura proposta (14 capítulos, 5 partes + apêndices)

**Parte I — Fundamentos**
1. Bem-vindo ao AutoML (o que é, por que existe, o problema CASH, histórico)
2. Aprendizado de máquina em 1 capítulo (dados, modelos, overfitting, validação cruzada, métricas)
3. Pipelines e pré-processamento (Pipeline, ColumnTransformer, vazamento de dados)

**Parte II — Otimização de hiperparâmetros**
4. Busca em grade e aleatória
5. Otimização bayesiana (processos gaussianos, TPE, SMAC, Optuna)
6. Métodos multifidelidade (Successive Halving, Hyperband, BOHB)
7. Métodos evolutivos e programação genética (TPOT)

**Parte III — Automação do pipeline**
8. Engenharia e seleção automática de atributos
9. Meta-aprendizado, warm-start e ensembles (Auto-sklearn)
10. Ferramentas de AutoML na prática (FLAML, Auto-sklearn, TPOT, H2O, AutoGluon, PyCaret)

**Parte IV — AutoML avançado**
11. Busca de arquiteturas neurais (NAS)
12. AutoML para séries temporais, texto e imagens

**Parte V — AutoML responsável e em produção**
13. Explicabilidade, ética, vieses e limitações
14. MLOps e projeto integrador final

**Apêndices:** A gabarito · B glossário · C guia rápido · D soluções dos exercícios práticos · E soluções dos projetos · Referências

Cada capítulo: objetivos, texto com analogias do cotidiano, figuras, 30 questões de múltipla escolha, 20 discursivas, exercícios práticos em Python (scikit-learn) e projeto do capítulo. Projeto integrador no Cap. 14.

## Restrições verificadas neste ambiente
- Sites externos (UCI, Kaggle, OpenML, JMLR, NeurIPS, Crossref) estão bloqueados; só o PyPI responde.
  - Exercícios executados usam bases UCI embutidas no scikit-learn (Iris, Wine, Breast Cancer Wisconsin, Digits) e dados simulados.
  - Kaggle/UCI externos aparecem apenas em "Para ir além", marcados como não testados.
- Referências NÃO podem ser conferidas online aqui. Entrarão só obras que conheço com segurança (autores, ano, veículo), marcadas para conferência final (DOI/páginas) por você.
- Instalado e executável: scikit-learn, Optuna, FLAML, XGBoost, LightGBM, SHAP.
- Não instaláveis aqui: TPOT, Auto-sklearn (Python 3.13), AutoGluon, H2O, PyCaret. Seus códigos aparecem no livro, mas sem saída executada, e isso será dito no texto.
