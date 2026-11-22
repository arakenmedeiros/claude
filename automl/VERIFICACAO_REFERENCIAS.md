# Registro de verificação das referências

Método: cada referência foi buscada com WebSearch (e Consensus, quando útil). Só entrou na lista com título, autores,
veículo e ano confirmados. Páginas, volumes e anos só aparecem quando a busca os confirmou; o que não foi
confirmado foi **omitido** (nunca preenchido de memória). Os sites das editoras e periódicos são inacessíveis
do ambiente, então a conferência final de DOI/páginas nos itens abaixo cabe ao autor.

## Correção feita pela verificação
- Kohavi (1995): páginas corretas são 1137-1145 (e não 1137-1143).

## Itens com dado omitido por não confirmado (conferir no site da editora/periódico)
| Referência | Dado pendente |
|---|---|
| Akiba et al. (Optuna, KDD 2019) | páginas |
| Bergmeir e Benítez (Information Sciences, 2012) | volume e páginas |
| Bischl et al. (WIREs DMKD, 2023) | lista completa de autores e nº do artigo (e1484) |
| Caruana et al. (ICML 2004) | páginas |
| Hutter, Hoos e Leyton-Brown (SMAC, LION 2011) | páginas |
| Lorena et al. (ACM CSUR 52(5), art. 107) | ano (fontes divergem: 2019 ou 2020) |
| Lundberg e Lee (NIPS 2017) | páginas |
| Ke et al. (LightGBM, NIPS 2017) | páginas |
| Thornton et al. (Auto-WEKA) | veículo (KDD 2013?) e páginas; confirmado título, autores, ano e arXiv 1208.3719 |
| Vanschoren et al. (OpenML) | ano (fontes divergem: 2013 ou 2014) |
| Zoph et al. (CVPR 2018) | páginas |
| LeDell e Poirier (H2O AutoML) | coautoria de Poirier (a busca só confirmou LeDell) |
| Erickson et al. (AutoGluon-Tabular) | lista completa de autores; confirmado como artigo do workshop AutoML do ICML 2020 e preprint arXiv 2003.06505 |
| Wang et al. (FLAML, MLSys 2021) | páginas |
| Schwartz et al. (Green AI) | ordem dos autores (fontes divergem) |

## Observações
- Livros (Hastie et al., Goodfellow et al., Rasmussen e Williams, Hyndman e Athanasopoulos, Hutter et al., Brazdil et al.): editora e ano confirmados por catálogos; local de publicação de Hyndman e Athanasopoulos não confirmado.
- Referências de preprint sem versão em periódico/evento NÃO foram incluídas.

## Revisão da edição (após o parecer)
Itens corrigidos ou completados (conferidos por busca):
- **Thornton et al. (Auto-WEKA):** acrescentados o evento (KDD 2013, 19.ª edição), as páginas (847-855) e a pré-publicação em arXiv (1208.3719). O DOI da ACM **não** foi confirmado e, por isso, não consta.
- **Vanschoren et al. (OpenML):** acrescentado o ano (2013; SIGKDD Explorations 15(2): 49-60).
- **Lorena et al.:** acrescentados o ano (2019) e o DOI (10.1145/3347711).
- **Bergstra e Bengio (2012)** e **Feurer et al. (2022):** acrescentadas as URLs oficiais do JMLR.

Referências **novas**, todas conferidas por busca (título, autores, veículo e ano):
Li et al. (2020, ASHA; MLSys, p. 230-246); Guo et al. (2017, ICML, PMLR 70, p. 1321-1330); Angelopoulos e Bates (2023, FnT-ML 16(4), p. 494-591, DOI 10.1561/2200000101); Vovk, Gammerman e Shafer (2005, Springer, DOI 10.1007/b106715); Pargent et al. (2022, Computational Statistics 37(5), p. 2671-2692, DOI 10.1007/s00180-022-01207-6); He e Garcia (2009, IEEE TKDE 21(9), p. 1263-1284, DOI 10.1109/TKDE.2008.239); Saito e Rehmsmeier (2015, PLoS ONE 10(3), e0118432, DOI 10.1371/journal.pone.0118432); Zhang e Zhou (2014, IEEE TKDE 26(8), p. 1819-1837, DOI 10.1109/TKDE.2013.39); Read et al. (2011, Machine Learning 85(3), p. 333-359, DOI 10.1007/s10994-011-5256-5); Gelbart, Snoek e Adams (2014, UAI; as **páginas** foram **omitidas**, pois as fontes consultadas divergem); Hollmann et al. (2023, TabPFN, ICLR); Lewis et al. (2020, RAG, NeurIPS 33; as páginas não foram confirmadas e foram omitidas).

Sobre DOIs e URLs: foram incluídos **apenas** os que a busca confirmou. As demais referências não trazem DOI/URL; a padronização completa depende de conferência nos sites das editoras, que não são acessíveis do ambiente de elaboração.

TPOT: a documentação oficial confirma que a versão 1.0 incorporou o TPOT2 e mudou a interface; o exemplo do livro foi identificado como da série 0.x (0.12.x) e **não foi executado**.
