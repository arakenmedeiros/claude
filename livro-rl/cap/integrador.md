@part: Parte V — Projeto Integrador
@banner: 14
# Uma plataforma de ofertas para reter clientes
@obj: Formular um problema real de decisão sequencial como MDP, com estados, ações e recompensas justificados.
@obj: Construir um simulador a partir de dados, validá-lo e quantificar sua incerteza.
@obj: Treinar e comparar agentes (DQN, PPO) com baselines simples e regras de negócio.
@obj: Testar a robustez das conclusões a hipóteses não mensuráveis nos dados.
@obj: Planejar a validação no mundo real (experimento randomizado e avaliação off-policy).
@obj: Redigir um relatório técnico com limitações, riscos éticos e recomendações.

## 1. O desafio

Uma loja virtual (a da base **Online Retail**, UCI) quer **reter clientes** que estão se afastando. A cada mês, pode enviar **ofertas** (nenhuma, cupom de 10% ou cupom de 20%). Ofertas custam dinheiro e **podem** aumentar as compras, principalmente de quem está "esfriando". Qual **política de ofertas** maximiza a receita líquida de longo prazo?

Este projeto reúne **todos os capítulos**:

| Etapa | Capítulos | Entrega |
|---|---|---|
| Formular o MDP e os estados dos clientes | 1, 3 | Definição e justificativa |
| Aproximar valores sem discretizar | 7 | Valor de vida por regressão |
| Simulador do cliente (modelo de mundo) | 11 | Modelo validado, com incerteza |
| Agentes DQN e PPO | 8, 9, 10 | Políticas treinadas |
| Baselines e programação dinâmica | 4, 6 | Comparação com regras e DP |
| Robustez e avaliação offline | 12 | Sensibilidade e plano de teste A/B |
| Segurança, ética e comunicação | 13 | Análise de riscos e relatório |

:::atencao Uma regra de ouro deste projeto
Os dados **não registram nenhuma oferta enviada**. O efeito dos cupons é **desconhecido**. Todo resultado sobre cupons vale **sob hipóteses** que você declarará e variará. A conclusão honesta é uma **recomendação condicional** e um **plano para medir o efeito de verdade**.
:::

## 2. Preparação dos dados

1. Gere o painel mensal cliente × mês com as características de recência, frequência e gasto (código do Capítulo 3; arquivo `painel_clientes.csv`).
2. Separe **clientes** em treino (70%) e teste (30%), por **cliente**, não por linha.
3. Descreva os dados: número de clientes, meses, distribuição do gasto, valores ausentes.

## 3. O modelo de mundo

Reaproveite a ideia do Capítulo 11: um regressor que, dado (recência, frequência, gasto), prevê o **próximo estado** e o **gasto do mês seguinte**. Para o projeto:

- Treine com os clientes de treino; avalie no teste, por horizonte.
- Meça a **incerteza** com um conjunto de modelos.
- Reporte até que horizonte o simulador é confiável. **Limite os episódios a esse horizonte.**

## 4. O ambiente de decisão

Crie um ambiente do Gymnasium (Apêndice F) com:

- **Estado:** recência, frequência e gasto (normalizados) e o mês do episódio.
- **Ações:** 0 (nada), 1 (cupom de 10%), 2 (cupom de 20%).
- **Efeito hipotético:** o gasto do mês seguinte é multiplicado por (1 + u), com u maior para clientes com recência alta (em risco), e a oferta também **reduz a recência** (o cliente "volta"). Os parâmetros do efeito (40% e 60% de aumento máximo de gasto para os cupons de 10% e 20%, por exemplo) são **hipóteses explícitas**, escolhidas para que o problema seja interessante: valem a pena para clientes em risco, mas não para quem já compra bem.
- **Recompensa:** receita do mês, menos o desconto concedido, menos um custo fixo por cupom.
- **Restrição de orçamento:** penalidade por exceder um limite de cupons por episódio.
- **Horizonte:** 6 a 12 meses, conforme a validação do simulador.

## 5. Código de partida

```python
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold

painel = pd.read_csv("painel_clientes.csv").sort_values(["cliente", "t"])
for c in ["rec", "freq", "mon"]:
    painel[c + "_prox"] = painel.groupby("cliente")[c].shift(-1)
d = painel.dropna(subset=["rec_prox", "gasto_prox"]).copy()
X = d[["rec", "freq", "mon"]].to_numpy(); Y = d[["rec_prox", "freq_prox", "mon_prox", "gasto_prox"]].to_numpy()
grupos = d["cliente"].to_numpy(); idx_tr, idx_te = next(GroupKFold(n_splits=3).split(X, groups=grupos))
mundo = ExtraTreesRegressor(n_estimators=40, min_samples_leaf=5, random_state=0, n_jobs=1).fit(X[idx_tr], Y[idx_tr])
residuos = Y[idx_tr] - mundo.predict(X[idx_tr])

class ClienteEnv(gym.Env):
    """Um cliente por episódio. Ações: 0 = nada, 1 = cupom 10%, 2 = cupom 20% (efeitos HIPOTÉTICOS)."""
    DESCONTO = np.array([0.0, 0.10, 0.20]); EFEITO_BASE = np.array([0.0, 0.40, 0.60]); CUSTO_FIXO = 1.0
    def __init__(self, X_inicio, efeito_escala=1.0, horizonte=6, orcamento=3, semente=0):
        super().__init__(); self.Xi, self.esc, self.H, self.orc = X_inicio, efeito_escala, horizonte, orcamento
        self.action_space = spaces.Discrete(3); self.observation_space = spaces.Box(0.0, 10.0, (4,), np.float32)
        self.rng = np.random.default_rng(semente)
    def _obs(self): return np.array([self.s[0] / 6 * 10, min(self.s[1], 3) / 3 * 10, min(self.s[2], 10) / 10 * 10, self.t / self.H * 10], np.float32)
    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self.s = self.Xi[self.np_random.integers(len(self.Xi))].astype(float).copy(); self.t = 0; self.usados = 0
        return self._obs(), {}
    def step(self, a):
        p = mundo.predict(self.s.reshape(1, -1))[0] + residuos[self.np_random.integers(len(residuos))]
        u = self.esc * self.EFEITO_BASE[a] * (0.3 + 0.7 * min(self.s[0], 2) / 2)            # efeito maior em quem está "esfriando"
        gasto = max(p[3], 0.0) * (1 + u); r = gasto * (1 - self.DESCONTO[a]) - (self.CUSTO_FIXO if a > 0 else 0.0)
        self.usados += int(a > 0); r -= 5.0 * max(0, self.usados - self.orc) * int(a > 0)  # penalidade acima do orçamento
        p[0] = p[0] * (1 - min(0.9, 2.0 * u))                                             # a oferta também "reativa" o cliente (reduz a recência)
        self.s = np.array([max(p[0], 0), max(p[1], 0), max(p[2], 0)]); self.t += 1
        return self._obs(), float(r), False, self.t >= self.H, {}

env_tr = ClienteEnv(X[idx_tr]); env_te = ClienteEnv(X[idx_te], semente=1)
print("estado inicial (treino):", env_tr.reset(seed=0)[0].round(2))
```

## 6. Baselines e agentes

Compare, no ambiente de **teste** (clientes que o simulador nunca viu), com **200 episódios** e várias sementes:

1. **Nunca ofertar.**
2. **Sempre cupom de 10%** e **sempre cupom de 20%** (respeitando o orçamento).
3. **Regra RFM:** cupom de 10% se recência ≥ 1; de 20% se recência ≥ 2.
4. **Programação dinâmica** sobre o MRP do Capítulo 4 (política do cupom por estado de K-Means).
5. **DQN** (Stable-Baselines3) treinado em `env_tr`.
6. **PPO** (Stable-Baselines3) treinado em `env_tr`.

```python
from stable_baselines3 import DQN, PPO

def avaliar_politica(env, politica, n=200, semente=0):
    ret = []
    for i in range(n):
        obs, _ = env.reset(seed=semente + i); total, fim = 0.0, False
        while not fim:
            obs, r, term, trunc, _ = env.step(int(politica(obs))); total += r; fim = term or trunc
        ret.append(total)
    return np.array(ret)

nunca = lambda obs: 0
regra = lambda obs: 2 if obs[0] / 10 * 6 >= 2 else (1 if obs[0] / 10 * 6 >= 1 else 0)
agente = DQN("MlpPolicy", env_tr, learning_rate=5e-4, buffer_size=30_000, learning_starts=500, batch_size=64,
             target_update_interval=250, exploration_fraction=0.3, policy_kwargs=dict(net_arch=[64, 64]), verbose=0, seed=0)
agente.learn(total_timesteps=5_000)
for nome, pol in [("nunca ofertar", nunca), ("regra RFM", regra), ("DQN", lambda o: agente.predict(o, deterministic=True)[0])]:
    r = avaliar_politica(env_te, pol, n=50)
    print(f"{nome:14s} | receita líquida por cliente em 6 meses: média = {r.mean():7.1f} (EP = {r.std() / np.sqrt(len(r)):.1f})")
```

## 7. Robustez e sensibilidade

Como o efeito do cupom é uma hipótese, avalie **cada política treinada sob a hipótese A** em ambientes com **outros efeitos** (por exemplo, `efeito_escala` ∈ {0; 0,5; 1; 2}). Use o `ParameterGrid` do scikit-learn. Responda:

- Em que faixa de efeito cada política vale a pena? Há uma política **robusta**, que não perde dinheiro mesmo se o efeito for zero?
- Qual o **efeito mínimo** para que ofertar compense o custo?
- O que acontece se o simulador estiver errado? Repita com os modelos individuais do conjunto.

```python
from sklearn.model_selection import ParameterGrid

linhas = []
for p in ParameterGrid({"efeito_escala": [0.0, 0.5, 1.0, 2.0]}):
    amb = ClienteEnv(X[idx_te], efeito_escala=p["efeito_escala"], semente=3)
    for nome, pol in [("nunca", nunca), ("regra RFM", regra), ("DQN", lambda o: agente.predict(o, deterministic=True)[0])]:
        linhas.append({**p, "política": nome, "receita": avaliar_politica(amb, pol, n=40).mean()})
print(pd.DataFrame(linhas).pivot(index="efeito_escala", columns="política", values="receita").round(1))
```

## 8. Do simulador ao mundo real

Nenhuma conclusão do simulador pode substituir um **experimento**. Planeje, por escrito:

1. **Teste A/B randomizado:** divida clientes aleatoriamente em grupos (sem oferta, 10%, 20%, política do agente). Defina a métrica principal (receita líquida em 6 meses), a duração, o tamanho de amostra necessário e os critérios de parada.
2. **Registro de propensões:** guarde a probabilidade com que cada oferta foi escolhida, para **avaliação off-policy** (IPS e DR, Capítulo 12) em dados futuros.
3. **Implantação gradual** com limites de orçamento e monitoramento.
4. **Atualização:** como o modelo será retreinado e como detectar mudanças nos clientes (deslocamento de distribuição).

## 9. Segurança, ética e comunicação

Responda no relatório:

- Que **riscos** a política traz? (por exemplo, treinar clientes a esperar cupons, tratamento desigual por país ou por perfil, uso de dados pessoais)
- Há **vieses** nos dados (por exemplo, clientes do Reino Unido dominando a base)?
- Quem pode **contestar** uma decisão? Que **supervisão humana** existe?
- O que **não** é possível concluir com estes dados?

## 10. Entrega e avaliação

**Entregáveis:** (1) *notebook* reproduzível, com sementes fixas e versões das bibliotecas; (2) relatório de 6 a 10 páginas; (3) resumo executivo de uma página para a gerência, **sem jargão**.

| Critério | Peso | O que se espera |
|---|---|---|
| Formulação do problema (MDP) | 15% | Estados, ações, recompensas e hipóteses claros e justificados |
| Simulador e validação | 20% | Validação por cliente, erro por horizonte, incerteza |
| Agentes e baselines | 20% | Comparação justa, várias sementes, intervalos de confiança |
| Robustez e sensibilidade | 15% | Análise do efeito dos cupons e do erro do simulador |
| Plano de validação real | 10% | A/B, propensões, avaliação off-policy |
| Ética e limitações | 10% | Riscos, vieses, o que não se pode concluir |
| Comunicação e reprodutibilidade | 10% | Clareza, gráficos, código organizado |

:::dica Dicas finais
- **Comece simples:** primeiro os baselines; depois, o DQN. Se um agente complexo não vence uma regra simples, **isso é um resultado**.
- **Desconfie dos bons resultados.** Procure vazamento, falhas do simulador e sorte de semente.
- **Escreva as limitações antes das conclusões.**
- **Peça revisão** de um colega: explicar o projeto a alguém é o melhor teste de entendimento.
:::

**Parabéns por chegar até aqui!** Você percorreu o caminho dos bandits aos modelos de linguagem alinhados por preferências. O restante é prática: escolha um problema que importe para você e aplique o que aprendeu, com cuidado, curiosidade e honestidade intelectual.
