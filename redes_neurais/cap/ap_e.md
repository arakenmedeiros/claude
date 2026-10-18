# Apêndice E — Instalação, reprodução e versões

## E.1 Ambiente

Os experimentos foram executados em Python 3 com as versões abaixo (arquivo `codigo/requirements.txt`):

| Pacote | Versão |
|---|---|
| numpy | 2.5.3 |
| scipy | 1.18.1 |
| scikit-learn | 1.9.1 |
| matplotlib | 3.11.2 |
| pandas | 3.0.5 |
| nbformat | 5.11.1 |

## E.2 Instalação

```python
# no terminal, dentro da pasta codigo/ do livro:
#   python3 -m venv .venv
#   source .venv/bin/activate        (Windows: .venv\Scripts\activate)
#   pip install -r requirements.txt
#   python3 testes_rnbook.py         (deve terminar com "TODOS OS TESTES OK")
```

## E.3 Pacote `rnbook`

O pacote `codigo/rnbook` contém a infraestrutura didática do livro: `MLP` (rede de múltiplas camadas com `forward`, `grad`, `jacobiana`), `otimizadores` (gradiente descendente, Quickprop, RPROP e variantes, Levenberg–Marquardt, delta-delta, DBD, EDBD, Gauss–Newton), `gradcheck` (verificação por diferenças finitas) e `ativacoes`. Para usar os scripts, execute-os a partir da pasta `codigo/` (ou defina `PYTHONPATH=codigo`).

## E.4 Reprodutibilidade

- Toda aleatoriedade usa sementes explícitas (`random_state`, `default_rng(seed)`, argumento `semente` da `MLP`); as sementes de cada experimento estão no texto e nos scripts.
- Tempos de execução dependem da máquina e da biblioteca BLAS; **não** são reproduzíveis exatamente. Contagens de iterações e perdas devem coincidir, a menos de diferenças numéricas de ponto flutuante entre plataformas.
- Mudanças de versão das bibliotecas podem alterar resultados do scikit-learn (por exemplo, inicialização e critérios de parada de `MLPClassifier`).

## E.5 O que não foi executado

Os trechos que baixam bases externas (UCI, Kaggle) e os procedimentos opcionais indicados como "para ir além" **não foram executados** na preparação do livro. Consulte também `PENDENCIAS.md`.
