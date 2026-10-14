# Apêndice J — Bases de dados: UCI e Kaggle
As práticas e projetos do livro usam **bases públicas**. Este apêndice mostra como obtê-las e quais cuidados tomar.

## J.1 UCI Machine Learning Repository

O **UCI Machine Learning Repository** (DUA; GRAFF, 2019) reúne centenas de bases clássicas, com documentação e referência dos autores originais. O pacote `ucimlrepo` baixa os dados diretamente pelo **identificador (id)** da base.

```python
from ucimlrepo import fetch_ucirepo

base = fetch_ucirepo(id=73)              # Mushroom
X = base.data.features                   # DataFrame com as características
y = base.data.targets                    # DataFrame com o alvo (se houver)
original = base.data.original            # tabela completa, tal como publicada
print(base.metadata["name"] if hasattr(base, "metadata") and "name" in base.metadata else "")
print(X.shape, list(X.columns)[:5])
```

| Base | id (UCI) | Descrição | Usada em |
|---|---|---|---|
| **Mushroom** | 73 | 22 características de cogumelos; comestível ou venenoso | Cap. 1, 2 |
| **Bank Marketing** | 222 | Campanhas de telemarketing de um banco português; o cliente contratou o depósito? (MORO; CORTEZ; RITA, 2014) | Cap. 2, 9, 12 |
| **Online Retail** | 352 | Compras de uma loja virtual do Reino Unido (CHEN; SAIN; GUO, 2012) | Cap. 3 a 8, 11, projeto final |
| **Wine Quality** | 186 | Propriedades físico-químicas de vinhos e nota de qualidade (CORTEZ et al., 2009) | Cap. 10, 13 |

:::atencao Os ids e as colunas podem mudar
Se `fetch_ucirepo(id=...)` falhar ou trouxer outros nomes de colunas, abra a página da base no site do UCI e confira o identificador e o dicionário de variáveis. Os códigos do livro foram escritos de forma tolerante (por exemplo, usam `.iloc[:, 0]` para o alvo), mas nomes de colunas específicos podem exigir ajuste.
:::

## J.2 Kaggle

O **Kaggle** é uma plataforma com milhares de bases, competições e notebooks. Para baixar dados por código:

1. Crie uma conta e gere um **token de API** (Configurações → API → *Create New Token*).
2. Instale e use o pacote `kagglehub`.

```python-nr
import kagglehub, pandas as pd, os

pasta = kagglehub.dataset_download("blastchar/telco-customer-churn")   # baixa e devolve o caminho
print(os.listdir(pasta))
df = pd.read_csv(os.path.join(pasta, os.listdir(pasta)[0]))
print(df.shape)
```

| Base (Kaggle) | Identificador | Uso sugerido |
|---|---|---|
| Telco Customer Churn | `blastchar/telco-customer-churn` | Retenção de clientes (Cap. 9, extensão) |
| E-Commerce Data (Online Retail) | `carrie1/ecommerce-data` | Alternativa à base UCI |
| Mushroom Classification | `uciml/mushroom-classification` | Alternativa à base UCI |
| Wine Quality (tinto) | `uciml/red-wine-quality-cortez-et-al-2009` | Alternativa à base UCI |
| Credit Card Fraud Detection | `mlg-ulb/creditcardfraud` | Decisão com classes raras (Cap. 12, extensão) |

:::atencao Verifique antes de usar
Os identificadores do Kaggle (os "slugs") e os arquivos podem mudar, e algumas bases exigem aceitar regras. Os trechos que usam o Kaggle não puderam ser executados no ambiente de verificação do livro (que não tem credenciais); por isso aparecem marcados como "não executados". Abra a página da base, confira o nome e o formato dos arquivos e adapte o código.
:::

## J.3 Boas práticas com dados

- **Cite a fonte.** Todas as bases têm autores e licenças; inclua a referência no seu relatório.
- **Leia o dicionário de dados.** O que significa cada coluna? Há valores especiais (como `-1` ou `"unknown"`)?
- **Cuide da privacidade.** Dados de pessoas exigem cuidado, mesmo quando anonimizados.
- **Desconfie de causalidade.** Dados observacionais mostram o que aconteceu, não o que aconteceria se a decisão fosse outra. Esse é o tema do Capítulo 12.
- **Divida os dados corretamente.** Separe treino e teste **antes** de ajustar qualquer transformação, e divida por **grupo** (cliente) quando houver várias linhas por pessoa.
- **Registre as versões.** Salve a data do download e os parâmetros de pré-processamento.

## J.4 Quando os dados não são suficientes

O AR pede **interação**. Bases estáticas servem para:

- **Construir simuladores** (modelar a resposta do ambiente com aprendizado supervisionado);
- **Avaliar políticas offline** (Capítulo 12);
- **Estimar cadeias e modelos** (Capítulos 3, 4 e 11).

Nunca confunda o resultado de um simulador construído por você com o desempenho no mundo real.
