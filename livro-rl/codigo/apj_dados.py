"""
Livro: Aprendizado por Reforço — Apêndice J — Bases de dados: UCI e Kaggle
Autor do livro: Araken de Medeiros Santos

Os blocos abaixo seguem a ordem do texto e devem ser executados em sequência (como células de um notebook).
Os trechos marcados como 'não executado' (por exemplo, downloads do Kaggle) aparecem comentados.
"""
import warnings; warnings.filterwarnings('ignore')

# ======================================================================
# Bloco 1
# ======================================================================
from ucimlrepo import fetch_ucirepo

base = fetch_ucirepo(id=73)              # Mushroom
X = base.data.features                   # DataFrame com as características
y = base.data.targets                    # DataFrame com o alvo (se houver)
original = base.data.original            # tabela completa, tal como publicada
print(base.metadata["name"] if hasattr(base, "metadata") and "name" in base.metadata else "")
print(X.shape, list(X.columns)[:5])

# ======================================================================
# Bloco 2 (trecho não executado: requer conta/download externo)
# ======================================================================
# import kagglehub, pandas as pd, os
# 
# pasta = kagglehub.dataset_download("blastchar/telco-customer-churn")   # baixa e devolve o caminho
# print(os.listdir(pasta))
# df = pd.read_csv(os.path.join(pasta, os.listdir(pasta)[0]))
# print(df.shape)

