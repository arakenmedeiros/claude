# Códigos do livro "Aprendizado Não Supervisionado"

Esta pasta traz um caderno Jupyter por capítulo (`capNN.ipynb`) com os blocos de código do texto, na ordem em que aparecem, e o arquivo `requirements.txt` com as versões das bibliotecas usadas na elaboração do livro.

## Como executar

```bash
python -m venv .venv
source .venv/bin/activate            # no Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

- Python 3.13. Os resultados numéricos podem variar um pouco em versões diferentes.
- Os blocos usam conjuntos de dados embutidos no scikit-learn ou dados simulados, com sementes fixas; não é preciso baixar arquivos.
- Os exercícios práticos e os projetos têm solução no Apêndice D e no Apêndice E do livro.
- O Capítulo 12 usa o PyTorch; o Capítulo 13 usa o NetworkX (somente na seção sobre comunidades).
