const { criarDeck } = require("./lib");
(async () => {
  const r = await criarDeck({ capNum: 3, capTitulo: "K-means: o clássico do agrupamento", subtitulo: "Do bairro das pizzarias ao cotovelo", autor: "Araken de Medeiros Santos", arquivo: "/tmp/teste_slides.pptx", slides: [
    { tipo: "capa", nota: "Abertura" },
    { tipo: "objetivos", itens: ["Explicar o **K-means** com a analogia das pizzarias", "Executar o algoritmo de Lloyd passo a passo", "Escolher k com cotovelo e silhueta"] },
    { tipo: "analogia", titulo: "Imagine 3 pizzarias", historia: "Você vai abrir **3 pizzarias** numa cidade. Cada morador pede na mais próxima. Onde colocá-las?", frase: "Pizzaria = centroide. Bairro = grupo.", fig: "f3_passos" },
    { tipo: "conceito", titulo: "O algoritmo em 4 passos", pontos: ["**Inicializar:** escolha k centros", "**Atribuir:** cada ponto ao centro mais próximo", "**Atualizar:** centro vira a média", "**Repetir** até estabilizar"], fig: "f3_voronoi", frase: "Atribuir, mover, repetir." },
    { tipo: "conceito", titulo: "Quando o K-means falha", pontos: ["**Formas** não esféricas", "**Tamanhos** muito diferentes", "**Outliers** puxam a média"], icone: "FaExclamationTriangle", frase: "K-means impõe bolas." },
    { tipo: "passos", titulo: "Passo a passo", passos: [{ t: "Inicializar", d: "k centros" }, { t: "Atribuir", d: "ponto ao centro mais próximo" }, { t: "Atualizar", d: "média de cada grupo" }], frase: "Convergiu: ninguém muda de grupo." },
    { tipo: "comparar", titulo: "K-means × Hierárquico", colunas: [{ t: "K-means", itens: ["Precisa de **k**", "Rápido", "Grupos esféricos"] }, { t: "Hierárquico", itens: ["Não precisa de k", "Lento", "Depende da ligação"] }], frase: "Escolha pelo problema." },
    { tipo: "tabela", titulo: "Variantes", cab: ["Variante", "Muda o quê?", "Quando usar"], linhas: [["Mini-Batch", "Usa lotes", "Dados enormes"], ["K-medoids", "Centro é ponto real", "Outliers"]], larg: [2.5, 3, 3.5] },
    { tipo: "formula", titulo: "O que ele minimiza", formula: "J = Σ ‖x − μ‖²", explicacao: ["**Inércia:** soma dos quadrados", "Quanto **menor**, mais compactos"], frase: "Compacto é bom." },
    { tipo: "codigo", titulo: "K-means em 6 linhas", codigo: "from sklearn.cluster import KMeans\nkm = KMeans(n_clusters=3, n_init=10)\nrot = km.fit_predict(X)\nprint(km.inertia_)", dicas: ["`n_clusters`: quantos grupos", "`fit_predict`: treina e devolve o grupo"] },
    { tipo: "numeros", titulo: "Em números", stats: [{ n: "k = 5", r: "melhor silhueta" }, { n: "0,70", r: "silhueta máxima" }, { n: "10x", r: "mais rápido (Mini-Batch)" }], frase: "Números contam a história." },
    { tipo: "memorize", cartoes: [{ p: "O que é centroide?", r: "A **média** do grupo." }, { p: "k-means++?", r: "Espalha os centros iniciais." }, { p: "Cotovelo?", r: "Onde a inércia desacelera." }, { p: "Falha onde?", r: "Luas e anéis." }] },
    { tipo: "quiz", p: "O centroide é calculado como:", ops: ["A mediana", "A média", "O ponto mais distante", "Um ponto sorteado"], certa: 1, expl: "É a média (por isso k-means)." },
    { tipo: "pratica", titulo: "Mão na massa", missao: ["Gere dados com make_blobs", "Varie k de 2 a 9", "Calcule silhueta"], entrega: { itens: ["Tabela k × silhueta", "Escolha justificada"] } },
    { tipo: "divisor", numero: "02", titulo: "Parte prática" },
    { tipo: "fim", pontos: ["K-means: atribuir e mover", "Minimiza a inércia", "k-means++ ajuda"], proximo: "Próximo: hierárquico" },
  ] });
  console.log(r);
})();
