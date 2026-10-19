# Apêndice H — Instalação e execução

Este apêndice explica como preparar o computador para rodar os códigos do livro, como usar o pacote didático `llmdidatico`, como (opcionalmente) ligar serviços e modelos reais e como reproduzir o próprio livro. Tudo o que está aqui foi escrito com o mesmo cuidado do restante: onde algo **não foi testado** no ambiente de elaboração, isso está dito.

:::fonte
**O que foi e o que não foi testado.** As versões abaixo são as **instaladas no ambiente em que os blocos executados do livro foram rodados** (Python 3.13.16, Linux). Os comandos de instalação com `pip` e `venv` são os padrões da linguagem. O uso no Google Colab, no Windows e no macOS **não foi testado** por nós; as instruções para esses casos seguem a prática comum e podem exigir ajustes. Versões mudam: confira sempre a documentação oficial.
:::

## H.1 O que você precisa

- **Python 3** (o livro usou a versão 3.13.16; versões próximas devem funcionar, mas **não foram verificadas**).
- Um terminal e cerca de 1 GB de espaço livre para as bibliotecas (valor aproximado; depende do sistema).
- **Nenhuma GPU** e **nenhuma conexão com a internet** são necessárias para os códigos executados do livro, que usam apenas NumPy, scikit-learn e o pacote didático. Internet só é necessária para instalar as bibliotecas e, opcionalmente, para serviços e modelos reais.

## H.2 Ambiente virtual e bibliotecas

Um **ambiente virtual** é uma pasta com as bibliotecas do projeto, isolada do resto do computador. Na raiz do repositório:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows (PowerShell): .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

O arquivo `requirements.txt` fixa as versões **exatas** instaladas na elaboração do livro:

| Biblioteca | Versão | Para que serve no livro |
| --- | --- | --- |
| numpy | 2.5.3 | arrays, álgebra linear, todos os modelos didáticos |
| scipy | 1.18.1 | estatística (testes, distribuições) |
| scikit-learn | 1.9.1 | TF-IDF, regressão logística, métricas, conjuntos embutidos |
| matplotlib | 3.11.2 | figuras |
| pandas | 3.0.5 | tabelas |
| pydantic | 2.13.5 | validação de saídas e de argumentos de ferramentas |
| anthropic | 1.15.0 | SDK de Python da Anthropic (apenas código `python-ni`) |
| transformers | 5.20.1 | biblioteca de modelos (instalada **sem pesos baixados**) |

Para conferir o que está instalado:

```bash
pip list | grep -i -E "numpy|scipy|scikit-learn|matplotlib|pandas|pydantic|anthropic|transformers"
python --version
```

As bibliotecas `torch`, `peft`, `bitsandbytes` e `sentence-transformers` **não** estavam instaladas no ambiente de elaboração. Os trechos do livro que as usam são `python-ni` e **não foram executados**.

## H.3 O pacote didático llmdidatico

O pacote `llmdidatico` está em `codigo/comum/llmdidatico/` e **não é instalado com `pip`**: basta colocá-lo no caminho de busca do Python (`PYTHONPATH`). Ele contém corpora sintéticos, a base fictícia de 60 documentos da Universidade Federal Aurora com 56 consultas, modelos de brinquedo em NumPy e simuladores. **Nada nele é um LLM.**

```bash
# Linux e macOS, a partir da raiz do repositório
export PYTHONPATH="$PWD/codigo/comum"
python -c "from llmdidatico import DOCS, CONSULTAS; print(len(DOCS), len(CONSULTAS))"
```

No Windows (PowerShell): `$env:PYTHONPATH = "$PWD\codigo\comum"`. Se a última linha acima imprimir o número de documentos e de consultas, o pacote está acessível.

## H.4 Rodando os códigos de cada capítulo

Cada capítulo tem uma pasta `codigo/capNN/` (NN de 01 a 18) com três arquivos gerados a partir do livro:

- `capNN_texto.py`: os blocos `python` e `python+` do texto, em ordem; cada `# %%` marca uma célula (funciona em VS Code, Spyder e Jupyter com `jupytext`, ou simplesmente como um script).
- `capNN_questoes.py`: os códigos de referência das questões discursivas de programação.
- `capNN_praticos.py`: as soluções dos exercícios práticos e do projeto do capítulo.

```bash
export PYTHONPATH="$PWD/codigo/comum"
export MPLBACKEND=Agg                       # figuras sem janela
python codigo/cap08/cap08_texto.py
python codigo/cap08/cap08_praticos.py
```

Os blocos `python-ni` (não executados) **não** estão nesses arquivos. As figuras do livro são geradas por `codigo/figuras/figs_capNN.py` e gravadas em `figs/`.

:::pratica
**Dica.** Altere **um** parâmetro por vez (uma semente, um limiar, um tamanho de fragmento) e observe o que muda. Com sementes fixas, a saída deve ser a mesma em cada execução na mesma versão das bibliotecas; versões diferentes podem alterar dígitos finais, e isso não é erro.
:::

## H.5 Google Colab e CPU

- **CPU.** É o modo de referência: todos os códigos executados do livro foram pensados para CPU.
- **Colab.** Em um notebook do Colab, envie a pasta `codigo/comum/llmdidatico` (ou clone o repositório), acrescente-a ao caminho com `import sys; sys.path.insert(0, "/content/<pasta>/codigo/comum")` e instale o que faltar com `pip`. O Colab já traz NumPy, SciPy, scikit-learn, matplotlib e pandas, **em versões que podem ser diferentes das do `requirements.txt`**.
- **GPU T4.** Uma GPU do tipo T4 (disponibilidade e limites variam conforme o plano e o momento; confira no serviço) é **opcional** e só faz diferença para os exercícios `python-ni` com modelos reais, como o ajuste com `transformers` e `peft` do Capítulo 12 e os modelos multimodais do Capítulo 13. Esses códigos **não foram executados** por nós.

## H.6 Uso opcional de APIs (sem chave no código)

Os códigos de API do livro são `python-ni`. Para tentá-los:

1. Crie uma conta e uma chave **no site do provedor** e leia a documentação oficial atual (preços, limites e nomes de modelos mudam).
2. Guarde a chave em **variável de ambiente**, nunca no código nem no repositório. O SDK da Anthropic lê, por padrão, a variável `ANTHROPIC_API_KEY`; confira a documentação da versão instalada.
3. O nome do modelo vem de outra variável, também definida fora do código (o livro usa `LLM_MODELO`).

```bash
read -s -p "Cole a chave (não aparece na tela): " ANTHROPIC_API_KEY && export ANTHROPIC_API_KEY
export LLM_MODELO="<nome-do-modelo-conforme-a-documentação-oficial>"
```

Antes de qualquer chamada: defina **limite de gasto** na conta do provedor, use `max_tokens` e tempo limite, e **não envie dados pessoais** sem base legal (Capítulos 9 e 17). Se uma chave for exposta, **revogue-a imediatamente**.

## H.7 Uso opcional de modelos locais

Modelos locais exigem **pesos obtidos por você**, que **não acompanham o livro**. Antes de baixar, leia a **licença do modelo específico e da versão**: pesos disponíveis não equivalem a modelo aberto (Capítulo 9). Há duas rotas, ambas `python-ni` no livro:

- **Biblioteca `transformers`**: carrega modelo e tokenizador de uma pasta ou do repositório do Hugging Face. Exige instalar também um backend de tensores (como `torch`), que **não estava instalado** no ambiente de elaboração.
- **Servidor local** (motores de inferência com API HTTP): o adaptador HTTP do Capítulo 9 conversa com o servidor; a porta e o caminho dependem do motor escolhido. Veja a documentação do motor.

Sem GPU, prefira modelos pequenos; meça tempo e memória antes de prometer qualquer coisa a alguém (Capítulo 15).

## H.8 Solução de problemas

| Sintoma | Causa provável | O que fazer |
| --- | --- | --- |
| `ModuleNotFoundError: llmdidatico` | `PYTHONPATH` não definido neste terminal | exporte `PYTHONPATH` apontando para `codigo/comum` (H.3); a variável vale só para o terminal atual |
| `ModuleNotFoundError` de outra biblioteca | ambiente virtual não ativado ou `pip install` não executado | ative o `.venv` e rode `pip install -r requirements.txt` |
| números ligeiramente diferentes dos do livro | versões diferentes das bibliotecas ou da plataforma | instale as versões do `requirements.txt`; diferenças nos dígitos finais são esperadas |
| figura não aparece, ou erro de interface gráfica | ambiente sem tela | defina `MPLBACKEND=Agg` e salve a figura com `savefig` |
| `KeyError: LLM_MODELO` | variável de ambiente ausente (código `python-ni`) | defina a variável (H.6); sem ela, o código não deve rodar |
| erro de autenticação ou de limite do provedor | chave ausente, inválida ou sem saldo; limite de taxa | confira a conta e a documentação; use recuo exponencial com jitter (Capítulo 9) |
| falta de memória ao carregar um modelo | modelo grande demais para a máquina | use modelo menor ou quantizado; reveja as contas do Capítulo 15 |
| o código `python-ni` falha | era ilustrativo e **não foi testado** | confira a documentação da versão instalada; os nomes de funções e parâmetros podem ter mudado |
| `pip` não encontra uma versão fixada | versão retirada ou incompatível com seu Python | instale a mais próxima e registre a diferença |

## H.9 Reproduzindo o livro (opcional)

Reproduzir o **DOCX** do livro é um passo à parte e **não é necessário para estudar**. O pipeline é: capítulos em Markdown (`cap/`), exercícios (`cap/exNN.txt`), práticos (`cap/pr/`), figuras (`codigo/figuras/`) e o montador `build/build.js`. Além do Python acima, ele usa:

- **Node.js** (testado com a versão 22.22.0) e os pacotes `docx` (9.7.1, instalado globalmente no ambiente de elaboração) e `katex` (0.19.0, em `build/node_modules`);
- **Playwright** 1.56.1 com Chromium, que `build/eqs.js` usa para desenhar as equações;
- **LibreOffice** (24.2.7.2) e as ferramentas `pdfinfo` e `pdftotext` (poppler), usadas para medir as páginas do sumário.

Os scripts em `build/` têm **caminhos absolutos do ambiente de elaboração** (`/home/user/claude/llm`, `/opt/node22/...`); ajuste-os se o repositório estiver em outro lugar. Os comandos principais estão no Apêndice L.
