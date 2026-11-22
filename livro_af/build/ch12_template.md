@banner: 12
# Do Laboratório ao Mundo Real: Assincronia, Participação, Esquecimento e Deriva
@obj: Explicar a diferença entre o aprendizado federado síncrono e o assíncrono e simular o FedAsync e o FedBuff com atrasos.
@obj: Reconhecer o viés de participação e comparar políticas de seleção de clientes.
@obj: Avaliar um modelo federado por clientes conhecidos e novos, distinguindo desempenho médio e pior caso.
@obj: Explicar por que a normalização deve ficar local quando os atributos têm "estilos" diferentes (FedBN).
@obj: Distinguir apagar dados, parar de participar e remover a influência de um cliente, e comparar o desaprendizado exato e o aproximado.
@obj: Usar pseudorrótulos e máscaras de rótulos quando há poucos rótulos ou rótulos diferentes em cada cliente.
@obj: Detectar a deriva de conceito e escolher uma estratégia de adaptação.

## 12.1 O mundo real não espera a rodada terminar

Até aqui, o aprendizado federado foi uma coreografia bem ensaiada. Em cada rodada, o servidor sorteava os clientes, todos treinavam, todos respondiam, e a média era feita. Os dados eram fixos. Os rótulos existiam em todos os clientes. Ninguém saía, ninguém entrava, ninguém pedia para ser esquecido.

Na vida real, a coreografia vira um **baile de rua**.

- Alguns clientes são **lentos** ou somem no meio da rodada.
- Quem participa não é um sorteio justo: **só aparece quem está com o aparelho carregando, na rede Wi-Fi e ocioso**.
- Ao avaliar o modelo, é preciso saber como ele se sai com clientes que **nunca participaram**.
- Cada cliente tem o seu "sotaque": as imagens, os sensores e os textos têm **estilos** diferentes.
- Uma pessoa exerce o direito de **ser esquecida**, e o modelo precisa perder a influência dela.
- Muitos clientes têm **poucos rótulos**, ou rótulos diferentes dos vizinhos.
- Com o tempo, o mundo **muda**, e o que o modelo aprendeu envelhece.

Este capítulo trata desses oito problemas, um por seção. Para cada um, há uma intuição, um experimento e um aviso sobre os limites.

![f12_mapa](Figura 12.1 — Os problemas do mundo real tratados neste capítulo e o que cada um muda no treino federado.)

:::cotidiano No seu dia a dia
Pense em um grupo de estudos online. Alguns colegas respondem em segundos, outros só à noite (**assincronia**). Quem fala mais acaba definindo o rumo da conversa (**viés de participação**). Alguém pede para apagar suas mensagens do material final (**esquecimento**). E, a cada semestre, a matéria muda (**deriva**).
:::

## 12.2 Aprendizado federado assíncrono

### 12.2.1 O problema do mais lento

No FedAvg **síncrono**, o servidor espera todos os clientes sorteados terminarem. Se um deles demora seis vezes mais, a rodada inteira demora seis vezes mais. Esse cliente é chamado de ***straggler*** (retardatário). É como uma fila de caixa de supermercado em que todos só saem quando o mais lento termina de passar as compras.

No aprendizado **assíncrono**, ninguém espera. Cada cliente recebe o modelo atual, treina no seu ritmo e **devolve o resultado quando terminar**. O servidor incorpora cada contribuição assim que ela chega.

Surge um novo problema: o **atraso** (*staleness*). Quando um cliente lento entrega o resultado, o modelo do servidor já mudou várias vezes. A atualização foi calculada sobre uma **versão velha** do modelo. Se o servidor a aplicar sem cuidado, pode puxar o modelo para trás.

![f12_assincrono](Figura 12.2 — No modo síncrono, a rodada espera o cliente mais lento. No assíncrono, cada resposta chega a qualquer momento, mas pode ter sido calculada sobre uma versão velha do modelo.)

### 12.2.2 Duas soluções: FedAsync e FedBuff

O **FedAsync** (Xie; Koyejo; Gupta, 2019) mistura cada resposta ao modelo na hora que ela chega, com um peso que **diminui com o atraso**:

$$ W ← (1 − a) · W + a · W_k,    a = α · (1 + atraso)^(−0,5) $$

Quanto mais velha a resposta, menos ela pesa.

O **FedBuff** (Nguyen et al., 2022) guarda as respostas em um **buffer**. Só quando chegam *K* atualizações, o servidor aplica a média delas. Cada atualização é dividida por √(1 + atraso). O buffer é útil por dois motivos: a média de várias respostas é **mais estável** que uma só, e o servidor pode somar o buffer com **agregação segura** (Capítulo 8) ou adicionar ruído de privacidade diferencial por buffer, o que não é possível com mistura individual.

:::saibamais Saiba mais
O **atraso** é medido em **versões do servidor**: se o cliente começou na versão 12 e entrega quando o servidor está na 15, o atraso é 3. Quando o servidor processa muitas respostas por unidade de tempo, o atraso cresce. Por isso, o FedBuff usa um *K* pequeno quando há poucos clientes e maior quando há muitos.
:::

### 12.2.3 O código

Daqui em diante, o capítulo usa uma biblioteca nova, **`avancado.py`**, construída aos poucos. Cada bloco rotulado como "parte" deve ser **acrescentado ao final** do arquivo. O primeiro bloco traz as importações.

Para comparar os métodos, simulamos o tempo com **tempo virtual**: cada cliente tem um tempo médio de treino (1 unidade para os rápidos, 6 para os lentos), com 20% de variação aleatória.

{{lib:@header,simular_sincrono,simular_assincrono}}

### 12.2.4 O experimento

Vinte clientes: 14 rápidos e 6 lentos. Para tornar o caso mais realista (e mais perigoso), os clientes **lentos concentram os dígitos 0, 1 e 2**: a velocidade está **correlacionada** com os dados. Cada método roda por 60 unidades de tempo, com três sementes. Medimos o tempo até atingir 90% de acurácia global e a acurácia nos dados dos clientes rápidos e dos lentos.

{{exp:e1}}

![f12_assinc_res](Figura 12.3 — Acurácia global ao longo do tempo (à esquerda) e acurácia nos dados dos clientes rápidos e lentos ao final (à direita).)

O que aprendemos:

- O **FedBuff** chegou aos 90% em bem menos tempo que o síncrono: enquanto os lentos terminam, os rápidos continuam contribuindo.
- O **FedAsync** foi **mais lento** que o síncrono para chegar aos 90% neste cenário. A mistura imediata de respostas individuais deixa o modelo oscilando, sobretudo quando os dados são diferentes.
- Há um **preço de justiça**. Nos dois métodos assíncronos, a acurácia nos dados dos clientes **lentos** é menor que no síncrono, porque suas contribuições chegam menos vezes e atrasadas. O síncrono espera os lentos, e por isso os atende melhor.

:::atencao Atenção
Assincronia **ganha tempo**, mas pode **perder justiça**: os clientes lentos (celulares antigos, redes ruins) ficam sub-representados, e talvez sejam justamente os que mais precisam do modelo. Meça a acurácia **por grupo de velocidade**, e não só a global.
:::

:::memorize Memorize
Síncrono: espera pelo mais lento. Assíncrono: ninguém espera, mas há **atraso**. FedAsync mistura na hora, com peso menor para respostas velhas. FedBuff junta *K* respostas e as aplica juntas, e é mais compatível com agregação segura e privacidade diferencial.
:::

## 12.3 Seleção de clientes e viés de participação

### 12.3.1 Quem aparece não é uma amostra aleatória

O FedAvg do Capítulo 3 sorteava clientes de modo **uniforme**. Isso supõe que qualquer cliente pode ser chamado a qualquer hora. Na prática, o servidor só consegue chamar quem está **disponível**: com bateria, na rede Wi-Fi, ocioso, em certo fuso horário.

Isso cria um **viés de participação**. Se a disponibilidade está relacionada aos dados (por exemplo, quem recarrega o telefone à noite usa outros aplicativos), o modelo é treinado com uma **fatia enviesada** da população. E quem fica de fora raramente é bem atendido.

Para a seleção, há três ideias principais.

- **Medir** quem participa (um registro de participação) e comparar com a população esperada.
- **Corrigir por probabilidade**: se um cliente só aparece 5% das vezes, o peso dele quando aparece deve ser maior, na proporção inversa da sua disponibilidade (a mesma ideia das pesquisas de opinião, que dão mais peso a quem é raro na amostra).
- **Escolher com critério**: o ***power-of-choice*** (Cho; Wang; Joshi, 2022) sorteia alguns candidatos disponíveis e fica com os que têm **maior perda local**, isto é, os que o modelo atende pior. Isso acelera o aprendizado e reduz o viés, mas custa avaliar a perda em todos os candidatos.

:::saibamais Saiba mais
Outra linha considera os **recursos dos clientes**: escolhe quem consegue terminar a rodada no prazo (Nishio; Yonetani, 2019). Isso reduz o tempo, mas aumenta o viés contra os dispositivos fracos. A escolha do critério de seleção é, portanto, **também uma escolha de justiça**.
:::

### 12.3.2 O código

{{lib:classe_dominante,fedavg_selecao}}

### 12.3.3 O experimento

Trinta clientes com dados muito diferentes (Dirichlet, α = 0,1). Quem tem os dígitos **0 a 4** como dominantes está disponível 95% das vezes. Quem tem os dígitos **5 a 9** está disponível apenas 5%. Em cada rodada, o servidor treina 4 clientes. Medimos a acurácia global, a média e o percentil 10 **por cliente** (inclusive os que quase nunca participam) e a acurácia na **pior classe**.

{{exp:e2}}

![f12_participacao](Figura 12.4 — Acurácia por dígito sob quatro políticas de seleção. O viés de disponibilidade derruba os dígitos 5 a 9.)

O que aprendemos:

- Com a disponibilidade enviesada, a acurácia na **pior classe** despenca (de cerca de 0,86 para 0,57), embora a acurácia global caia pouco. A média **esconde** quem ficou para trás.
- A **ponderação por disponibilidade** recupera parte do dano (pior classe em torno de 0,67).
- O ***power-of-choice*** recupera mais (pior classe em torno de 0,76), pois chama justamente os clientes que o modelo atende pior.
- Nenhuma política recupera tudo: sem a participação **real** dos clientes raros, não há milagre.

:::atencao Atenção
Estimar a disponibilidade de cada cliente exige **registrar quem participa**, o que cria um dado sensível por si só (a rotina do aparelho de alguém). Registre de forma agregada e por tempo limitado.
:::

## 12.4 Avaliar de verdade: clientes conhecidos, novos e partições naturais

### 12.4.1 Duas perguntas diferentes

Quando dizemos "o modelo federado tem 95% de acurácia", precisamos perguntar: **de quem?** Yuan et al. (2022) propõem separar duas lacunas.

- **Lacuna de dados novos:** o modelo é pior em **dados novos** de clientes que participaram do treino?
- **Lacuna de participação:** o modelo é pior em **clientes que nunca participaram**, de distribuições talvez diferentes?

A segunda é a que importa para quem entrega o modelo a **novos usuários**: um novo hospital, um novo celular, uma nova filial.

### 12.4.2 Partições naturais e artificiais

Até aqui, repartimos os dígitos **artificialmente** (IID, *shards*, Dirichlet). Isso é útil, mas é uma simulação. Em **partições naturais**, o cliente é uma entidade real: o **escritor** de uma letra, o **usuário** de um teclado, o **hospital** de um exame. Conjuntos como o **LEAF** (Caldas et al., 2018), com o FEMNIST (caracteres por escritor) e textos de Shakespeare (por personagem), trazem esses clientes naturais.

Os conjuntos disponíveis dentro do scikit-learn **não** trazem identificadores de cliente. Por isso, usamos aqui um **substituto**: damos a cada cliente um **estilo** próprio (um ganho e um deslocamento na intensidade dos pixels, como o traço mais forte ou mais fraco de um escritor). Não é um conjunto natural, mas reproduz a **diferença de estilo** que partições naturais costumam ter. Em um projeto real, use dados naturais, como os do LEAF, do TensorFlow Federated ou do Flower Datasets.

{{lib:federacao_com_estilo,ajustar_padronizador}}

### 12.4.3 O experimento

Trinta clientes, dos quais **24 participam do treino** (conhecidos) e **6 nunca participam** (novos). O padronizador é aprendido **só com os clientes conhecidos** e aplicado aos novos, como ocorreria na vida real. Comparamos quatro cenários: sem estilo, estilo moderado, estilo forte e um cenário em que os **novos clientes vêm de uma população mais extrema** que a do treino.

{{exp:e4}}

![f12_estilos](Figura 12.5 — O mesmo dígito com estilos diferentes de clientes (à esquerda) e a acurácia em clientes conhecidos e novos (à direita).)

O que aprendemos:

- O estilo, sozinho, já **derruba** a acurácia e, sobretudo, o **percentil 10** e o **pior cliente**: com estilo forte, o pior cliente fica perto de 0,5 enquanto a média ainda parece razoável.
- Quando os novos clientes vêm da **mesma população** dos conhecidos, a lacuna de participação é pequena.
- Quando vêm de uma **população diferente**, a lacuna aparece: a média dos novos cai bem abaixo da dos conhecidos, e o pior novo fica abaixo de 0,5.

:::dica Dica
Ao relatar um modelo federado, apresente sempre: (1) média **por cliente**, e não só a global; (2) o **percentil 10** ou o **pior cliente**; (3) o desempenho em **clientes retidos** (nunca vistos no treino); (4) o tipo de partição (natural ou artificial).
:::

## 12.5 FedBN: a normalização que fica em casa

### 12.5.1 O sotaque dos atributos

O experimento anterior mostrou que o "estilo" dos atributos derruba o desempenho. Uma das causas é a **normalização**. Redes neurais modernas usam a **normalização em lote** (*batch normalization*, Ioffe; Szegedy, 2015): em cada camada, os valores são recentrados e reescalados com a média e a variância do lote, e depois ajustados por dois parâmetros aprendidos, a **escala** *g* e o **deslocamento** *b*.

Se os clientes têm estilos diferentes, **as médias e variâncias de cada cliente são diferentes**. Fazer a média dessas estatísticas e desses parâmetros entre clientes é como misturar duas receitas com forno diferente e esperar um bolo bom. Hsieh et al. (2020) mostraram que a normalização em lote é um dos pontos frágeis do treino com dados não-IID.

O **FedBN** (Li et al., 2021) propõe uma solução simples: **os parâmetros e as estatísticas da normalização não são enviados ao servidor**. Cada cliente mantém os seus. As demais camadas continuam sendo agregadas como no FedAvg.

| O que | No FedAvg | No FedBN |
|---|---|---|
| Pesos das camadas lineares (*W*) | Agregados | Agregados |
| Escala e deslocamento (*g*, *b*) | Agregados | **Locais** |
| Média e variância da normalização | Agregadas | **Locais** |

### 12.5.2 Uma rede com normalização, do zero

Para ver o efeito, escrevemos com NumPy uma rede de uma camada oculta com normalização em lote, **incluindo a retropropagação pela normalização**. (O gradiente foi conferido contra diferenças numéricas.)

{{lib:iniciar_mlp_bn,frente_bn,passo_bn,estatisticas_bn,fedbn_treinar}}

### 12.5.3 O experimento

Dez clientes IID nos rótulos, mas com **estilos diferentes** nos atributos (ou sem estilo, como controle). O padronizador é global e aprendido só com os dados de treino. Comparamos o FedAvg (tudo compartilhado) e o FedBN, avaliando cada cliente no seu teste local.

{{exp:e5}}

![f12_fedbn](Figura 12.6 — O que o FedBN mantém em casa (à esquerda) e o resultado com e sem estilos diferentes (à direita).)

O que aprendemos:

- **Sem estilos**, FedAvg e FedBN ficam equivalentes (o FedBN é até um pouco pior, pois personaliza sem necessidade).
- **Com estilos**, o FedAvg cai bastante, sobretudo no percentil 10, e o FedBN **mantém** o desempenho. Manter a normalização local absorve a diferença de estilo.

:::atencao Atenção
O FedBN é **personalização**: o cliente tem partes próprias do modelo. Portanto, um **cliente novo** precisa de dados para estimar as suas estatísticas locais. Se ele tem poucos exemplos, essas estatísticas ficam ruidosas. E, como em toda personalização, quem tem poucos dados pode se beneficiar menos.
:::

## 12.6 Desaprendizado federado

### 12.6.1 Esquecer é mais difícil que aprender

Um cliente pede: "removam os meus dados". A **LGPD** e o **GDPR** reconhecem direitos de eliminação, e há outros motivos: um cliente comprometido que precisa ser retirado, ou dados obtidos sem consentimento. Mas o que significa "esquecer" em um modelo federado? Há três coisas diferentes.

1. **Apagar os dados** do cliente. Fácil, mas o modelo **continua** com a influência deles.
2. **Parar de convocar o cliente** daqui para a frente. Também fácil, mas o que ele ensinou **continua no modelo**.
3. **Remover a influência** do cliente no modelo. É o **desaprendizado** (*unlearning*, Cao; Yang, 2015), e é o difícil.

O desaprendizado **exato** é fácil de definir: o resultado deve ser igual ao de um modelo **treinado sem aquele cliente**. A forma garantida de obtê-lo é **retreinar do zero** sem ele (ou, em arquiteturas especiais, retreinar apenas uma parte, como no SISA de Bourtoule et al., 2021). Mas retreinar é **caro**: refaz todas as rodadas, em todos os clientes.

Surgem então os métodos **aproximados**. O **FedEraser** (Liu et al., 2021) guarda no servidor o **histórico** das atualizações dos clientes. Depois de remover um cliente, reconstrói o modelo aproveitando as **normas** das atualizações antigas dos clientes restantes e usando, como **direção**, um treino curto de **calibração**. É mais barato que retreinar, mas é aproximado.

![f12_desaprendizado](Figura 12.7 — Quatro caminhos para "esquecer" um cliente e o resultado de cada um: acurácia no dígito que só ele possuía e custo em épocas de treino.)

### 12.6.2 O código

{{lib:fedavg_historico,sem_cliente,subtrair_contribuicao,calibrar_federaser}}

### 12.6.3 O experimento

Criamos um caso em que a influência é **mensurável**: **só o cliente 0 possui exemplos do dígito 9**. O modelo treinado com todos acerta o 9. O modelo **ideal**, treinado sem o cliente 0, **não pode** acertar o 9, pois nunca o viu. Comparamos cinco opções e medimos: a acurácia no dígito 9, a acurácia nos demais, a **perda nos dados do cliente removido** (quanto maior, mais o modelo "estranha" esses dados) e o custo em épocas de treino dos clientes.

{{exp:e3}}

O que aprendemos:

- **Só parar** de convocar o cliente por mais 10 rodadas **não esquece**: o modelo ainda acerta 87% dos 9, e a perda nos dados do cliente continua baixa. O conhecimento permanece.
- **Subtrair a contribuição** zera o 9, mas **danifica** o resto: a acurácia nos demais cai e a perda no cliente removido chega a 6,8, bem acima da do retreino (3,1). O modelo foi **puxado demais**.
- A **calibração no estilo FedEraser** chega perto do retreino (9 zerado, perda em 3,2, acurácia nos demais igual), com **cerca de um quinto do custo**.
- O **retreino exato** é o padrão de referência, e o mais caro.

:::atencao Atenção
**Desaprender não é uma garantia de apagamento.** Um método aproximado pode parecer funcionar em uma métrica (aqui, a acurácia no dígito 9) e deixar rastros em outra. Para **verificar**, use mais de uma medida: a perda nos dados removidos, a proximidade do modelo ao retreino exato, e ataques de pertencimento (Capítulo 7). Garantias **formais** de esquecimento exigem técnicas como a privacidade diferencial (Capítulo 8) ou o retreino exato. Além disso, o servidor que guarda o histórico de atualizações cria **um novo risco de privacidade** e custo de armazenamento.
:::

:::memorize Memorize
Apagar dados ≠ parar de convocar ≠ remover a influência. O padrão de referência é o **retreino sem o cliente**. Os métodos aproximados (como o FedEraser) trocam um pouco de exatidão por custo menor, e precisam ser **verificados**.
:::

## 12.7 Poucos rótulos: semissupervisionado e autossupervisionado federado

### 12.7.1 O problema

Rotular dados custa caro e exige especialistas. Em muitas federações, **só alguns clientes** têm rótulos (por exemplo, uma clínica com especialistas), e os demais só têm exemplos **sem rótulo**. Há duas famílias de ideias.

- **Semissupervisionado:** usar os dados sem rótulo para ajudar. A técnica mais simples é o **pseudorrótulo** (Lee, 2013): o modelo atual prevê o rótulo dos exemplos sem rótulo, e os exemplos em que ele tem **alta confiança** passam a ser usados no treino como se fossem rotulados. O **FedMatch** (Jeong et al., 2021) acrescenta uma perda de **consistência entre clientes**.
- **Autossupervisionado:** aprender uma **representação** dos dados sem usar rótulos (por exemplo, comparando duas visões do mesmo exemplo) e, depois, treinar um classificador leve com poucos rótulos. Em aprendizado federado, as representações são aprendidas com os dados sem rótulo de todos os clientes, e só então os rótulos escassos refinam o resultado. No Capítulo 11, vimos uma forma simples de aprendizado de representação sem rótulos: o PCA federado.

O perigo comum é a **propagação de erros**. Um pseudorrótulo errado vira dado de treino, ensina o modelo a errar do mesmo jeito, e gera mais pseudorrótulos errados. É o efeito de **confirmação**: o modelo confirma as próprias crenças.

### 12.7.2 O código

{{lib:fedavg_pseudorotulos}}

### 12.7.3 O experimento

Vinte clientes (Dirichlet, α = 0,5). **Apenas 3 têm rótulos.** Os demais passam a usar pseudorrótulos depois de 10 rodadas de aquecimento. Comparamos o treino **só com os rotulados**, três limiares de confiança e o teto (todos rotulados). Medimos a acurácia e a **precisão dos pseudorrótulos aceitos** (que só podemos calcular porque, no experimento, conhecemos os rótulos verdadeiros).

{{exp:e6}}

![f12_semi](Figura 12.8 — Acurácia e precisão dos pseudorrótulos para diferentes limiares de confiança.)

O que aprendemos:

- Os pseudorrótulos **não ajudaram** aqui. Com limiar alto, a precisão é altíssima (0,99), mas os exemplos aceitos são justamente aqueles que o modelo **já acertava**: não há informação nova. Com limiar baixo, entram mais erros (precisão de 0,88) e o ganho continua nulo.
- O teto, com todos os rótulos, mostra o que está em jogo: a diferença entre 0,87 e 0,96 é o que os rótulos faltantes valem.
- O resultado depende do cenário. Com modelos mais flexíveis, dados mais estruturados e perdas de consistência, o semissupervisionado costuma ajudar (Jeong et al., 2021). O ponto aqui é **cautelar**: pseudorrótulo não é mágica, e **não substitui** rótulos para classes que o modelo ainda não aprendeu.

:::dica Dica
Se você usar pseudorrótulos, **audite** uma amostra: peça a um especialista para rotular uma pequena fração dos exemplos aceitos e meça a precisão. Se ela estiver baixa, aumente o limiar ou pare.
:::

## 12.8 Multirrótulo e hierarquia: rótulos diferentes em cada cliente

### 12.8.1 Um exame, vários rótulos

Em problemas reais, um exemplo pode ter **vários rótulos ao mesmo tempo**: uma radiografia mostra "derrame pleural" **e** "cardiomegalia". É a **classificação multirrótulo** (Zhang; Zhou, 2014). O modelo tem **uma saída sigmoide por rótulo**, em vez de uma única saída softmax.

Na federação, aparece um problema novo: **cada instituição anota apenas os rótulos que lhe interessam**. Uma clínica anota só o rótulo A; outra, A e B; outra, só C. Isso é diferente de o rótulo ser "falso": ele está **ausente**.

Há um erro comum: tratar o rótulo **ausente** como **negativo** (valor 0). O modelo aprende que "A nunca existe" nos clientes que não anotam A. A solução é usar uma **máscara**: a perda só conta os rótulos **anotados**. E a agregação deve ser feita **por rótulo**, considerando só os clientes que anotam aquele rótulo.

Há também o problema da **hierarquia**: "pneumonia" é um tipo de "doença pulmonar", que é um tipo de "alteração torácica". Um modelo que prevê o filho (pneumonia) sem prever o pai (doença pulmonar) é **incoerente**. Há duas saídas: forçar a coerência na **inferência** (derivar o pai do filho) ou na **perda**.

### 12.8.2 O código

Para o experimento, derivamos três rótulos de cada dígito: "é par", "é maior ou igual a 5" e "é primo".

{{lib:rotulos_multi,sigmoide,treino_local_multi,fedavg_multi,metricas_multi}}

### 12.8.3 O experimento

Vinte clientes IID. Cada um anota **só 1 ou 2** dos três rótulos (o cliente 0 anota os três, e o cliente 1 só o rótulo "primo"). Comparamos o treino com rótulos completos (o teto), tratar o ausente como 0, a máscara com média global e a máscara com média **por rótulo**. Depois, testamos a coerência hierárquica: o dígito (filho) e a paridade (pai).

{{exp:e7}}

![f12_multirotulo](Figura 12.9 — Quais rótulos cada cliente anota (à esquerda) e o F1 de cada estratégia (à direita).)

O que aprendemos:

- Tratar o rótulo ausente como negativo é **catastrófico**: o F1 despenca (de 0,91 para 0,46 em "par").
- A **máscara** recupera quase todo o teto, e a média **por rótulo** é um pouco melhor que a global.
- Na hierarquia, os dois modelos treinados separadamente são **incoerentes** em cerca de 9% dos exemplos. **Derivar o pai do filho** corrige a incoerência por construção e ainda **melhora** a acurácia do pai (de 0,91 para 0,97), pois o filho tem mais informação.

:::memorize Memorize
Rótulo ausente **não** é rótulo negativo. Use **máscara** na perda e agregue **por rótulo**. Em hierarquias, garanta a **coerência** pai-filho na inferência ou na perda.
:::

## 12.9 Deriva de conceito: o mundo muda

### 12.9.1 O que é a deriva

Um modelo treinado ontem pode ser ruim hoje. A **deriva de conceito** (*concept drift*, Gama et al., 2014) é a mudança, com o tempo, da relação entre os atributos e o rótulo. Exemplos: um novo tipo de fraude, uma nova gíria, uma nova gripe, um novo protocolo clínico. Em aprendizado federado, ela é mais difícil de lidar porque:

- a deriva pode ocorrer **em momentos diferentes** em cada cliente (uma clínica adota o novo protocolo antes das outras);
- o servidor **não vê** os dados, só as perdas e as atualizações;
- o modelo global precisa **servir a conceitos diferentes ao mesmo tempo**.

Há dois cenários.

- **Deriva global:** todos os clientes mudam ao mesmo tempo. O modelo precisa se **readaptar**. Estratégias: continuar treinando (o modelo esquece aos poucos), ou **detectar** a deriva (a perda média sobe de repente) e **reiniciar** o modelo.
- **Deriva parcial (distribuída):** só uma parte dos clientes muda. Um único modelo global tenta servir a dois conceitos e falha. É a situação do **agrupamento de clientes** do Capítulo 6: o IFCA descobre os grupos. O **FedDrift** (Jothimurugesan et al., 2023) formaliza isso como um problema de agrupamento que varia no tempo.

### 12.9.2 O código

{{lib:fedavg_com_deriva}}

### 12.9.3 O experimento

Vinte clientes (Dirichlet, α = 1). Na rodada 25, os rótulos mudam: os dígitos "trocam de nome" por uma permutação. Primeiro, a **deriva global** (todos mudam): comparamos **continuar** e **reiniciar ao detectar**. Depois, a **deriva parcial** (só metade muda): comparamos um modelo único e o IFCA com *K* = 2.

{{exp:e8}}

![f12_deriva](Figura 12.10 — Acurácia por rodada com deriva global na rodada 25 (à esquerda) e acurácia nos dois conceitos com deriva parcial (à direita).)

O que aprendemos:

- O detector disparou **na própria rodada da deriva** (rodada 25), nas três sementes.
- **Reiniciar** recupera a acurácia (cerca de 0,87) em **uma rodada**, enquanto **continuar** leva uns 10 rodadas para voltar ao nível anterior.
- Na deriva **parcial**, um modelo único **fracassa nos dois conceitos** (cerca de 0,63 e 0,45). O IFCA, ao agrupar, serve bem os **dois** (cerca de 0,96 e 0,95).
- Reiniciar é boa ideia na deriva **global** e **péssima** na parcial: jogaria fora o que os clientes sem deriva ainda sabem.

:::atencao Atenção
Um detector de deriva baseado na perda média tem um **dilema**: sensível demais, dispara por ruído (falsos alarmes); insensível demais, demora. Calibre com dados históricos e use **janelas** de tempo. E lembre: deriva nos **rótulos** (o conceito mudou) é diferente de deriva nos **atributos** (a população mudou), e exigem respostas diferentes.
:::

## 12.10 Lista de verificação do mundo real

Antes de colocar uma federação em produção, pergunte:

- **Tempo:** a rodada espera o mais lento? Se não, como medir o atraso e o efeito sobre os lentos?
- **Participação:** quem participa de fato? A disponibilidade tem relação com os dados? Há ponderação ou seleção que compense?
- **Avaliação:** há desempenho **por cliente** e em **clientes retidos**? Os dados de teste são naturais ou artificiais?
- **Atributos:** os clientes têm "estilos" diferentes? As estatísticas de normalização podem ficar locais?
- **Esquecimento:** há um plano para atender pedidos de remoção? Que método, que custo, que verificação?
- **Rótulos:** quem anota o quê? Rótulo ausente está sendo tratado como ausente?
- **Tempo e mudança:** há detecção de deriva? Um modelo único serve a todos, ou é preciso agrupar?

## 12.11 Resumo do capítulo

:::memorize O essencial do Capítulo 12
- **Assíncrono:** ninguém espera, mas há **atraso**. FedAsync mistura na hora; **FedBuff** junta *K* respostas. O preço é a **justiça** com os lentos.
- **Seleção:** quem aparece não é amostra aleatória. Meça, **pondere** pela disponibilidade e considere o ***power-of-choice***.
- **Avaliação:** olhe **por cliente**, o **pior caso** e os **clientes novos**. A média esconde quem ficou para trás.
- **FedBN:** com estilos diferentes, deixe a **normalização local**.
- **Desaprendizado:** apagar dados ≠ parar de convocar ≠ remover a influência. O padrão é o **retreino**; os aproximados precisam ser **verificados**.
- **Rótulos escassos:** pseudorrótulos **não são mágica**. Rótulo **ausente** não é negativo: use **máscara** e agregue **por rótulo**.
- **Deriva:** detecte, **reinicie** se for global e **agrupe** se for parcial.
:::
