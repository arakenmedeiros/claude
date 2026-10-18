# Apêndice C — Guia rápido de decisões, ferramentas e códigos
## C.1 Mapa de decisão

| Se você quer… | Considere | Capítulo |
|---|---|---|
| Reproduzir um treino em outra máquina | Semente fixa, `requirements.txt`, ambiente virtual, contêiner | 1, 4, 7 |
| Saber se o projeto está pronto para produção | ML Test Score (28 itens, nota = mínimo dos grupos) | 2 |
| Garantir a qualidade dos dados de entrada | Esquema, validação com portão, quarentena | 3 |
| Identificar a versão dos dados | Hash SHA-256, manifesto, DVC | 3 |
| Evitar vazamento no pré-processamento | `Pipeline`, `ColumnTransformer`, `fit` só no treino | 4 |
| Registrar experimentos | MLflow Tracking | 5 |
| Trocar ou reverter o modelo sem mexer no código | Registro de modelos com apelido `champion` | 5 |
| Avaliar com classes raras | Precisão, revocação, F1, AUC-PR | 6 |
| Escolher o limiar de decisão | Custo dos erros, escolha na validação | 6 |
| Comparar modelos com rigor | Validação cruzada repetida, bootstrap, validação aninhada | 6 |
| Ajustar hiperparâmetros | Busca aleatória, Optuna, Hyperband | 6 |
| Servir previsões | Lote (padrão), tempo real (FastAPI), fluxo contínuo, embarcado | 7 |
| Rejeitar entradas inválidas | Contrato com Pydantic (HTTP 422) | 7 |
| Empacotar o ambiente | Contêiner (Docker) | 7 |
| Testar o sistema | `pytest`: código, dados e modelo; portão de qualidade | 8 |
| Automatizar a entrega | CI/CD; treinamento contínuo (CT) | 8 |
| Detectar mudança em um atributo | KS, PSI, Jensen-Shannon, qui-quadrado | 9 |
| Detectar mudança multivariada | Classificador de domínio (AUC) | 9 |
| Detectar piora com rótulos | DDM, ADWIN, métricas móveis | 9 |
| Lançar um modelo novo com segurança | Sombra, canário, teste A/B, reversão | 9 |
| Documentar um modelo | Cartão do modelo (*model card*) | 10 |
| Explicar um modelo | Permutação, PDP, LIME, SHAP | 10 |
| Auditar justiça | Taxa de seleção, revocação e precisão por grupo | 10 |
| Proteger privacidade | Minimização, pseudonimização, privacidade diferencial | 3, 10 |
| Evitar vazamento do futuro em atributos | Junção temporal (`merge_asof`), armazém de atributos | 10 |
| Executar tarefas em ordem | Orquestrador (DAG) | 10 |

## C.2 Classes, funções e comandos principais

| Tarefa | Ferramenta | Pontos de atenção |
|---|---|---|
| Pipeline | `sklearn.pipeline.Pipeline`, `make_pipeline` | Nomeie as etapas; use `fit` só no treino |
| Colunas mistas | `sklearn.compose.ColumnTransformer` | `OneHotEncoder(handle_unknown="ignore")` |
| Divisão por grupo | `GroupShuffleSplit`, `GroupKFold` | Evita o mesmo indivíduo nos dois lados |
| Série temporal | `TimeSeriesSplit` | Treino no passado, teste no futuro |
| Busca de hiperparâmetros | `GridSearchCV`, `RandomizedSearchCV` | Nunca use o teste |
| Calibração | `CalibratedClassifierCV`, `calibration_curve` | Compare Brier antes e depois |
| Explicação | `sklearn.inspection.permutation_importance`, `partial_dependence` | Importância não é causa |
| Salvar modelo | `joblib.dump` / `joblib.load` | Registre a versão do scikit-learn; só abra arquivos confiáveis |
| Rastrear | `mlflow.start_run`, `log_param`, `log_metric`, `log_model` | Registre hash dos dados |
| Registrar | `mlflow.register_model`, `set_registered_model_alias` | Carregue por `models:/nome@champion` |
| Contrato | `pydantic.BaseModel`, `Field(ge=..., le=...)` | Rejeita dados inválidos antes do modelo |
| API | `fastapi.FastAPI`, `TestClient` | Carregue o modelo uma vez (`lifespan`) |
| Testes | `pytest`, `parametrize`, `raises` | Fixe sementes; use tolerâncias |
| DAG | `graphlib.TopologicalSorter` | Dependências sem ciclos |

## C.3 Lista de verificação de prontidão para produção

- [ ] Dados validados por esquema; hash/versão registrados.
- [ ] Linha de base definida; métrica de negócio e custo dos erros claros.
- [ ] Pré-processamento dentro do pipeline; sem vazamento.
- [ ] Sementes e versões de bibliotecas fixadas; treino reproduzível.
- [ ] Execuções registradas; modelo no registro, com assinatura e apelido.
- [ ] Avaliação rigorosa: validação cruzada, incerteza, calibração, limiar e fatias.
- [ ] Contrato de entrada e saída; testes de código, dados e modelo.
- [ ] CI com portão de qualidade; imagem versionada; plano de reversão.
- [ ] Monitoramento de dados, previsões e desempenho; alertas e runbook.
- [ ] Cartão do modelo; avaliação de justiça e privacidade; responsáveis definidos.

## C.4 Valores de referência usados no livro

| Item | Valor | Observação |
|---|---|---|
| PSI | < 0,10 estável; 0,10 a 0,25 moderada; > 0,25 importante | Regra prática, não lei |
| Razão de impacto | 0,8 | "Regra dos quatro quintos": sinal de alerta, não veredito |
| Intervalo de confiança | 95% (percentis 2,5 e 97,5) | Bootstrap |
| DDM | alerta em mínimo + 2 desvios; deriva em mínimo + 3 | Gama et al. (2004) |
| ML Test Score | 0 / 0,5 / 1 ponto por item; nota = mínimo dos grupos | Breck et al. (2017) |

:::atencao Atenção
Todos os valores acima são **convenções**. Ajuste-os ao seu contexto, ao risco do problema e ao histórico de variação normal dos seus dados.
:::
