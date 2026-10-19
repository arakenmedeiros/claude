# Grandes Modelos de Linguagem — livro didático e material de apoio

Livro didático em português do Brasil, de **Araken de Medeiros Santos**, com 18 capítulos (o último com seis projetos integradores), apêndices, exercícios, projetos, código executável e slides. Público misto, sem conhecimentos prévios de aprendizado de máquina. Data de corte bibliográfico: **18 de outubro de 2026**.

## Aviso importante sobre o que este repositório é e não é

- Os experimentos do livro usam **corpora sintéticos em português** e o pacote didático `llmdidatico` (NumPy). **Nenhum experimento usou um LLM real.** Onde o texto mostra um "modelo", ele é um **simulador de regras** ou um modelo de brinquedo, declarado como tal.
- Os números das saídas descrevem esses simuladores e modelos pequenos, **não** o comportamento de modelos reais. Não os use para escolher produtos.
- Código que depende de serviços ou modelos reais está rotulado `python-ni` e **não foi executado**.
- Preços, versões, licenças e leis citados podem ter mudado: confira as fontes oficiais.

## Estrutura de pastas

| Caminho | Conteúdo |
| --- | --- |
| `cap/chNN.md` | texto dos capítulos 1 a 18 (o capítulo 18 reúne os seis projetos integradores) |
| `cap/exNN.txt` | questões do capítulo: 30 de múltipla escolha e 20 discursivas |
| `cap/pr/prNN.txt` | exercícios práticos e projeto de fim de capítulo, com soluções |
| `cap/referencias.txt` | referências, com status de verificação (V, C ou P) |
| `cap/apresentacao.md`, `cap/partes.txt` | apresentação e textos das partes |
| `cap/glossario.txt`, `simbolos.md`, `instalacao.md`, `prompts.md`, `checklists.md`, `matriz.md`, `metodo.md` | apêndices F a L |
| `codigo/comum/llmdidatico/` | pacote didático em NumPy (corpora sintéticos, base fictícia, modelos de brinquedo, simuladores, métricas) |
| `codigo/capNN/` | códigos extraídos do livro: `_texto.py`, `_questoes.py`, `_praticos.py` |
| `codigo/figuras/` | scripts que geram as figuras |
| `figs/` | figuras geradas (PNG) |
| `build/` | montador do livro (Node.js e Python), runners e verificadores; `code_out.json`, `ex_ok.json` e `pr_out.json` guardam as saídas reais das execuções |
| `slides/` | apresentações por capítulo |
| `BRIEF.md`, `SLIDES_BRIEF.md` | instruções internas de produção |
| `requirements.txt` | versões exatas das bibliotecas Python |

## Como começar a usar o código

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/codigo/comum" MPLBACKEND=Agg
python codigo/cap08/cap08_texto.py
```

O guia completo (Colab, CPU, APIs opcionais, modelos locais, solução de problemas) está no Apêndice H (`cap/instalacao.md`).

## Como reproduzir cada parte

Os scripts de `build/` usam o caminho absoluto `/home/user/claude/llm`; ajuste `ROOT` se o repositório estiver em outro lugar.

| Parte | Comando | Observação |
| --- | --- | --- |
| blocos de código de um capítulo | `python3 build/runcode.py N` | executa `python` e `python+`, grava as saídas em `build/code_out.json` e gera `codigo/capNN/capNN_texto.py` |
| questões de programação | `python3 build/runex.py N` | executa os blocos `CODE:` e grava `build/ex_ok.json` |
| exercícios práticos | `python3 build/runpr.py N` | grava `build/pr_out.json` |
| figuras | `python3 codigo/figuras/figs_capNN.py` | grava em `figs/` |
| verificação de formato | `python3 build/chk.py` | confere questões, referências e figuras citadas |
| livro em DOCX | `cd build && node eqs.js && node build.js ../livro.docx` | exige Node.js, `docx`, `katex`, Playwright; `build/make.sh` faz o processo completo com LibreOffice (veja o Apêndice L) |

## O que foi executado e o que não foi

**Executado, com saída real registrada** (contagens nos arquivos JSON de `build/`, na data do corte):

- 209 blocos de código `python`/`python+` dos capítulos 1 a 18 (`code_out.json`);
- 78 códigos de questões discursivas de programação, todos com resultado verdadeiro (`ex_ok.json`);
- 72 resultados de práticos: 54 exercícios e 18 projetos (`pr_out.json`).

**Não executado:** blocos `python-ni` (14 nos capítulos 1 a 18: adaptadores de API e de modelos locais, `transformers` e `peft`, modelos multimodais reais, servidores de inferência e, no capítulo 18, adaptadores de API real para gerador, agente e extração), qualquer chamada a serviço real, qualquer modelo real, qualquer uso de GPU.

## Licença e uso

**A definir pelo autor.** Até que uma licença seja escolhida, não presuma permissão de redistribuição ou de reuso além da leitura e da execução pessoal. Os dados do pacote `llmdidatico` são **fictícios ou sintéticos**; instituições, prazos, valores e pessoas citados nos exemplos não são reais (os CPFs gerados pelo pacote são números sorteados e não devem ser usados para outro fim).
