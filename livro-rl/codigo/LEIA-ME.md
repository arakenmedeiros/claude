# Códigos do livro "Aprendizado por Reforço"

Cada arquivo `chNN.py` reúne, na ordem do texto, os blocos de código do capítulo NN; `integrador.py` é o projeto integrador e `apX_*.py` são os apêndices.

## Como executar

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python ch02.py
```

- As versões das bibliotecas usadas na verificação estão em `requirements.txt`.
- Os capítulos que usam bases do UCI baixam os dados pela biblioteca `ucimlrepo` (é preciso acesso à internet).
- Os capítulos 3, 4, 5, 7, 11 e o projeto integrador usam arquivos gerados no Capítulo 3 (`painel_clientes.csv`, `P_clientes.npy`, `r_clientes.npy`): execute `ch03.py` primeiro, na mesma pasta.
- Os resultados variam entre execuções e entre versões das bibliotecas e do hardware (sementes não garantem reprodutibilidade completa).
