# Apêndice C — Guia rápido de algoritmos e códigos
## C.1 Mapa de decisão

| Se você quer… | Considere | Capítulo |
|---|---|---|
| Agrupar grandes bases com grupos esféricos | K-means, Mini-Batch K-means | 3 |
| Obter uma hierarquia de grupos | Aglomerativo (Ward, média) | 4 |
| Achar grupos de forma livre e ruído | DBSCAN, HDBSCAN, OPTICS | 5 |
| Grupos elípticos e probabilidades | Mistura gaussiana (GMM) | 6 |
| Separar formas complexas em bases médias | Agrupamento espectral | 6 |
| Avaliar agrupamentos | Silhueta, CH, DB, gap, ARI, NMI | 7 |
| Resumir e visualizar rapidamente | PCA | 8 |
| Separar fontes misturadas | ICA | 8 |
| Partes aditivas e tópicos | NMF | 8 |
| Mapas 2D de estruturas curvas | t-SNE, UMAP, Isomap | 9 |
| Itens que ocorrem juntos | Apriori, FP-Growth | 10 |
| Encontrar casos raros | Isolation Forest, LOF, One-Class SVM | 11 |
| Aprender representações | Autoencoder, VAE, contrastivo | 12 |

## C.2 Classes do scikit-learn e parâmetros principais

| Técnica | Classe ou função | Parâmetros-chave |
|---|---|---|
| K-means | sklearn.cluster.KMeans | n_clusters, n_init, init |
| Mini-Batch K-means | sklearn.cluster.MiniBatchKMeans | n_clusters, batch_size |
| Hierárquico | sklearn.cluster.AgglomerativeClustering | n_clusters, linkage, distance_threshold |
| DBSCAN | sklearn.cluster.DBSCAN | eps, min_samples, metric |
| OPTICS | sklearn.cluster.OPTICS | min_samples, xi, min_cluster_size |
| HDBSCAN | sklearn.cluster.HDBSCAN | min_cluster_size, min_samples |
| GMM | sklearn.mixture.GaussianMixture | n_components, covariance_type, n_init |
| Espectral | sklearn.cluster.SpectralClustering | n_clusters, affinity, n_neighbors |
| Silhueta | sklearn.metrics.silhouette_score | X, rótulos |
| ARI e NMI | sklearn.metrics.adjusted_rand_score, normalized_mutual_info_score | rótulos de referência e previstos |
| PCA | sklearn.decomposition.PCA | n_components, svd_solver |
| ICA | sklearn.decomposition.FastICA | n_components |
| NMF | sklearn.decomposition.NMF | n_components, init |
| Kernel PCA | sklearn.decomposition.KernelPCA | kernel, gamma |
| MDS | sklearn.manifold.MDS | n_components, dissimilarity |
| Isomap | sklearn.manifold.Isomap | n_neighbors, n_components |
| t-SNE | sklearn.manifold.TSNE | perplexity, init, learning_rate |
| UMAP | umap.UMAP | n_neighbors, min_dist, metric |
| Isolation Forest | sklearn.ensemble.IsolationForest | n_estimators, contamination |
| LOF | sklearn.neighbors.LocalOutlierFactor | n_neighbors, contamination, novelty |
| One-Class SVM | sklearn.svm.OneClassSVM | kernel, gamma, nu |
| Apriori e FP-Growth | mlxtend.frequent_patterns.apriori, fpgrowth | min_support |
| Regras | mlxtend.frequent_patterns.association_rules | metric, min_threshold |

## C.3 Modelo de código para um projeto completo

```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1) Dados (aqui, um exemplo embutido)
from sklearn.datasets import load_wine
X = load_wine(as_frame=True).data

# 2) Pré-processar
pre = Pipeline([("imputar", SimpleImputer(strategy="median")),
                ("escalar", StandardScaler())])
Xp = pre.fit_transform(X)

# 3) Testar vários k e avaliar
for k in range(2, 8):
    r = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Xp)
    print(k, round(silhouette_score(Xp, r), 3))

# 4) Escolher k, treinar e perfilar os grupos
rotulos = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(Xp)
print(X.assign(grupo=rotulos).groupby("grupo").mean().round(1).T)
```

## C.4 Lista de verificação final

:::memorize Antes de apresentar seus resultados
- Os dados foram limpos, e os atributos padronizados quando necessário?
- Existe tendência de agrupamento (Hopkins)?
- Testei mais de um valor de k ou de parâmetros, e mais de um algoritmo?
- Os grupos são estáveis em reamostragens?
- Os grupos têm perfil claro, nome e utilidade para decisão?
- Considerei riscos éticos, privacidade e viés?
- Documentei sementes, versões e escolhas?
:::
