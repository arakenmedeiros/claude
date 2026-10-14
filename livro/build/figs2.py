from fc import *
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, set_link_color_palette
from scipy.spatial.distance import pdist

def f4_taxonomia():
    nomes=["Cachorro","Lobo","Gato","Leão","Baleia","Golfinho","Águia","Pardal","Salmão","Tubarão"]
    # colunas: pelos, amamenta, quatro patas, voa, penas, bico, nada, escamas, garras, carnívoro
    M=np.array([[1,1,1,0,0,0,0,0,1,0],[1,1,1,0,0,0,0,0,1,1],[1,1,1,0,0,0,0,0,1,1],[1,1,1,0,0,0,0,0,1,1],
                [0,1,0,0,0,0,1,0,0,0],[0,1,0,0,0,0,1,0,0,1],[0,0,0,1,1,1,0,0,1,1],[0,0,0,1,1,1,0,0,0,0],
                [0,0,0,0,0,0,1,1,0,0],[0,0,0,0,0,0,1,1,0,1]],float)
    Z=linkage(pdist(M,"jaccard"),"average")
    fig,ax=plt.subplots(figsize=(7.2,3.8)); set_link_color_palette([IND,GRN,AMB,RED])
    dendrogram(Z,labels=nomes,ax=ax,color_threshold=0.8,above_threshold_color=GRY,leaf_rotation=40,leaf_font_size=9)
    ax.set_ylabel("Altura (dissimilaridade)"); ax.set_title("Dendrograma de animais por características compartilhadas")
    set_link_color_palette(None); save(fig,"f4_taxonomia")

def f4_dendro():
    rng=np.random.default_rng(3); X=np.r_[rng.normal([0,0],0.5,(3,2)),rng.normal([3,0.5],0.5,(3,2)),rng.normal([1.5,3.5],0.5,(2,2))]
    Z=linkage(X,"ward"); fig,ax=plt.subplots(figsize=(6.6,4.2))
    d=dendrogram(Z,ax=ax,labels=[f"P{i+1}" for i in range(8)],color_threshold=0,above_threshold_color=INK)
    H=Z[:,2].max(); ax.set_ylabel("Altura = distância da fusão"); ax.set_ylim(0,H*1.12)
    ax.text(1.02,0.07,"folhas = exemplos",transform=ax.transAxes,fontsize=8.5,color=IND,fontweight="bold",ha="right")
    ax.text(0.99,0.88,"fusão alta →\ngrupos diferentes",transform=ax.transAxes,fontsize=8.5,color=RED,fontweight="bold",ha="right")
    ax.text(0.5,0.30,"← fusão baixa:\nmuito parecidos",transform=ax.transAxes,fontsize=8.5,color=GRN,fontweight="bold",ha="left")
    save(fig,"f4_dendro")

def f4_corte():
    X,_=make_blobs(40,centers=4,cluster_std=0.9,random_state=7); Z=linkage(X,"ward"); h=Z[:,2]
    fig,ax=plt.subplots(figsize=(7.2,4)); set_link_color_palette(CL[:4])
    dendrogram(Z,ax=ax,color_threshold=h[-3]-0.01,above_threshold_color=GRY,no_labels=True)
    ax.axhline(h[-3]-0.5 if False else (h[-4]+h[-3])/2,c=GRN,ls="--",lw=2); ax.text(1,(h[-4]+h[-3])/2+1,"Corte A → 4 grupos",color=GRN,fontweight="bold")
    ax.axhline((h[-2]+h[-1])/2,c=RED,ls="--",lw=2); ax.text(1,(h[-2]+h[-1])/2+1,"Corte B → 2 grupos",color=RED,fontweight="bold")
    ax.set_ylabel("Altura"); ax.set_title("Cortar a árvore em alturas diferentes"); set_link_color_palette(None); save(fig,"f4_corte")

def f4_ligacoes():
    rng=np.random.default_rng(1)
    A=rng.normal([0,0],0.5,(60,2)); B=rng.normal([5,0],0.5,(60,2)); C=rng.normal([2.5,4],0.5,(60,2)); bridge=np.c_[np.linspace(0.8,4.2,12),rng.normal(0,0.08,12)]
    X=np.r_[A,B,C,bridge]
    fig,ax=plt.subplots(1,4,figsize=(11,2.9))
    for a,(m,t) in zip(ax,[("single","Simples"),("complete","Completa"),("average","Média"),("ward","Ward")]):
        l=AgglomerativeClustering(3,linkage=m).fit_predict(X); pts(a,X,[CL[i] for i in l],12); a.set_title(t,fontsize=10); clean(a,True)
    fig.suptitle("Mesmos dados, quatro ligações (com uma 'ponte' de pontos entre dois grupos)",fontsize=9.5,fontweight="bold",y=1.03)
    fig.tight_layout(); save(fig,"f4_ligacoes")

def f4_heatmap():
    rng=np.random.default_rng(5); n,m=36,14
    M=rng.normal(0,0.6,(n,m)); rb=np.repeat([0,1,2],12); cb=np.repeat([0,1,2],[5,5,4])
    for i in range(n):
        for j in range(m):
            if rb[i]==cb[j]: M[i,j]+=2.2
    p=rng.permutation(n); q=rng.permutation(m); M=M[p][:,q]
    Zr=linkage(M,"ward"); Zc=linkage(M.T,"ward")
    fig=plt.figure(figsize=(6.4,5.4)); ax1=fig.add_axes([0.02,0.08,0.14,0.72]); ax2=fig.add_axes([0.18,0.82,0.62,0.14]); ax3=fig.add_axes([0.18,0.08,0.62,0.72])
    dr=dendrogram(Zr,orientation="left",ax=ax1,no_labels=True,color_threshold=0,above_threshold_color=INK); dc=dendrogram(Zc,ax=ax2,no_labels=True,color_threshold=0,above_threshold_color=INK)
    for a in (ax1,ax2): a.axis("off")
    im=ax3.imshow(M[dr["leaves"][::-1]][:,dc["leaves"]],aspect="auto",cmap="viridis"); ax3.set_xlabel("Atributos (colunas agrupadas)"); ax3.set_ylabel("Exemplos (linhas agrupadas)"); ax3.yaxis.set_label_position("right")
    ax3.set_xticks([]); ax3.set_yticks([])
    cb=fig.add_axes([0.83,0.08,0.025,0.72]); fig.colorbar(im,cax=cb,label="valor")
    save(fig,"f4_heatmap",tight=False)

def f5_ilhas():
    rng=np.random.default_rng(2)
    fig,ax=plt.subplots(figsize=(7.2,3.8)); ax.set_facecolor(NAVY); fig.patch.set_facecolor(NAVY)
    cities=[((2,2),0.45,260),((6.5,2.8),0.6,380),((4.5,0.8),0.25,120),((8,0.6),0.3,150),((1.5,3.5),0.2,80)]
    for c,s,n in cities:
        P=rng.normal(c,s,(n,2)); ax.scatter(P[:,0],P[:,1],s=rng.uniform(2,10,n),c="#FDE68A",alpha=0.75,lw=0)
        ax.scatter(*c,s=1400*s,c="#FDE68A",alpha=0.05,lw=0)
    N=np.c_[rng.uniform(0,9,50),rng.uniform(0,4,50)]; ax.scatter(N[:,0],N[:,1],s=6,c="#F87171",alpha=0.85,lw=0)
    ax.set_xlim(0,9); ax.set_ylim(0,4); clean(ax)
    ax.text(0.2,3.75,"manchas densas = grupos",color="#FDE68A",fontsize=9,fontweight="bold"); ax.text(0.2,3.5,"luzes isoladas (vermelho) = ruído",color="#F87171",fontsize=9,fontweight="bold")
    fig.savefig(OUT+"f5_ilhas.png",bbox_inches="tight",facecolor=NAVY,pad_inches=0.1); plt.close(fig); print("fig f5_ilhas")

def f5_conceitos():
    X=np.array([[0,0],[0.5,0.3],[-0.4,0.5],[0.3,-0.5],[-0.6,-0.3],[0.9,-0.1],[1.2,0.5],[1.5,0.1],[-0.9,0.4],[0.1,1.0],[2.4,0.9],[2.1,-0.9],[4.0,0.0],[3.6,1.8],[-1.9,-1.1],[-2.4,1.5],[0.2,-1.9],[3.8,-1.6]])
    eps=0.85; ms=4
    D=np.linalg.norm(X[:,None]-X[None],axis=2); neigh=[np.where(D[i]<=eps)[0] for i in range(len(X))]
    core=np.array([len(n)>=ms for n in neigh]); border=np.array([(not core[i]) and any(core[j] for j in neigh[i]) for i in range(len(X))]); noise=~core&~border
    fig,ax=plt.subplots(figsize=(6.4,4.2))
    for i in np.where(core)[0]: ax.add_patch(Circle(X[i],eps,fc=IND,alpha=0.07,ec=IND,lw=0.8))
    ax.scatter(*X[core].T,c=IND,s=60,label="central",edgecolors="white"); ax.scatter(*X[border].T,c=AMB,s=70,label="borda",edgecolors="white"); ax.scatter(*X[noise].T,c=RED,s=70,marker="X",label="ruído",edgecolors="white")
    ax.set_aspect("equal"); ax.set_xlim(-3.3,5.2); ax.set_ylim(-2.8,2.9); ax.legend(loc="upper right",fontsize=8,title=f"eps = {eps}  |  min_samples = {ms}",title_fontsize=8); clean(ax,True)
    save(fig,"f5_conceitos")
    print("core",core.sum(),"border",border.sum(),"noise",noise.sum())

def f5_vs_kmeans():
    X=StandardScaler().fit_transform(make_moons(400,noise=0.07,random_state=0)[0])
    a=KMeans(2,n_init=10,random_state=0).fit_predict(X); b=DBSCAN(eps=0.3,min_samples=5).fit_predict(X)
    fig,ax=plt.subplots(1,2,figsize=(7.6,3.4))
    pts(ax[0],X,[CL[i] for i in a],14); ax[0].set_title("K-means (erra)"); clean(ax[0],True)
    pts(ax[1],X,[CL[i] if i>=0 else RED for i in b],14); ax[1].set_title("DBSCAN (acerta)"); clean(ax[1],True)
    save(fig,"f5_vs_kmeans")

def f5_kdist():
    X=StandardScaler().fit_transform(make_moons(400,noise=0.07,random_state=0)[0])
    d,_=NearestNeighbors(n_neighbors=5).fit(X).kneighbors(X); dk=np.sort(d[:,-1])
    fig,ax=plt.subplots(figsize=(6,3.5)); ax.plot(dk,c=IND,lw=2.4)
    ax.axhline(0.3,c=RED,ls="--"); ax.text(10,0.31,"joelho → eps ≈ 0,3",color=RED,fontweight="bold")
    ax.set_xlabel("Pontos (ordenados pela distância)"); ax.set_ylabel("Distância ao 5º vizinho"); ax.set_title("Gráfico de k-distâncias"); ax.grid(alpha=0.3)
    save(fig,"f5_kdist")

def f5_densidades():
    rng=np.random.default_rng(0)
    X=np.r_[rng.normal([0,0],0.25,(200,2)),rng.normal([4,3],1.0,(200,2))]
    fig,ax=plt.subplots(1,3,figsize=(10.4,3.3))
    for a,(e,t) in zip(ax,[(0.15,"eps pequeno: o grupo difuso\nvira ruído"),(0.5,"eps médio: bom para um,\nruim para o outro"),(1.3,"eps grande: tudo se funde")]):
        l=DBSCAN(eps=e,min_samples=8).fit_predict(X); pts(a,X,[CL[i%8] if i>=0 else GRY for i in l],10); a.set_title(t,fontsize=9); clean(a,True)
    fig.tight_layout(); save(fig,"f5_densidades")

def f5_optics():
    rng=np.random.default_rng(0)
    X=np.r_[rng.normal([0,0],0.3,(150,2)),rng.normal([4,4],1.0,(150,2)),rng.normal([-4,4],0.6,(150,2))]
    o=OPTICS(min_samples=10,xi=0.05,min_cluster_size=0.1).fit(X)
    r=o.reachability_[o.ordering_]; l=o.labels_[o.ordering_]
    fig,ax=plt.subplots(1,2,figsize=(10,3.4),gridspec_kw={"width_ratios":[1.6,1]})
    r2=np.where(np.isinf(r),np.nan,r)
    for k in sorted(set(l)):
        idx=np.where(l==k)[0]; ax[0].bar(idx,r2[idx],width=1,color=CL[k%8] if k>=0 else GRY)
    ax[0].set_title("Gráfico de alcançabilidade: vales = grupos"); ax[0].set_xlabel("Pontos na ordem do OPTICS"); ax[0].set_ylabel("Distância de alcançabilidade")
    pts(ax[1],X,[CL[i%8] if i>=0 else GRY for i in o.labels_],10); ax[1].set_title("Grupos resultantes"); clean(ax[1],True)
    fig.tight_layout(); save(fig,"f5_optics")

def f6_gauss2d():
    fig,ax=plt.subplots(1,4,figsize=(10.4,2.8)); rng=np.random.default_rng(0)
    covs=[([[1,0],[0,1]],"Esférica\n(círculo)"),([[2.4,0],[0,0.5]],"Diagonal\n(alinhada aos eixos)"),([[1.6,1.2],[1.2,1.6]],"Correlação positiva\n(inclinada ↗)"),([[1.6,-1.2],[-1.2,1.6]],"Correlação negativa\n(inclinada ↘)")]
    for a,(c,t),col in zip(ax,covs,[IND,GRN,AMB,RED]):
        X=rng.multivariate_normal([0,0],c,250); pts(a,X,col,10)
        w,v=np.linalg.eigh(np.array(c)); ang=np.degrees(np.arctan2(v[1,1],v[0,1]))
        for k in (1,2): a.add_patch(Ellipse((0,0),2*k*np.sqrt(w[1]),2*k*np.sqrt(w[0]),angle=ang,fc="none",ec=INK,lw=1.4))
        a.set_xlim(-5,5); a.set_ylim(-5,5); a.set_aspect("equal"); a.set_title(t,fontsize=9); clean(a,True)
    fig.tight_layout(); save(fig,"f6_gauss2d")

def f6_mistura():
    rng=np.random.default_rng(1); x=np.r_[rng.normal(120,12,400),rng.normal(172,9,600)]
    fig,ax=plt.subplots(figsize=(6.4,3.5)); ax.hist(x,bins=45,density=True,color=MID,edgecolor="white")
    g=np.linspace(80,210,400); n=lambda m,s: np.exp(-0.5*((g-m)/s)**2)/(s*np.sqrt(2*np.pi))
    ax.plot(g,0.4*n(120,12),c=AMB,lw=2.4,label="componente 1 (crianças)"); ax.plot(g,0.6*n(172,9),c=IND,lw=2.4,label="componente 2 (adultos)")
    ax.plot(g,0.4*n(120,12)+0.6*n(172,9),c=INK,lw=2,ls="--",label="mistura (soma ponderada)")
    ax.set_xlabel("Altura (cm)"); ax.set_yticks([]); ax.legend(fontsize=8,frameon=False); ax.set_title("Duas corcovas = duas gaussianas somadas")
    save(fig,"f6_mistura")

def f6_em():
    X,_=make_blobs(400,centers=[[0,0],[3.5,1],[1.5,4]],cluster_std=[0.6,0.9,0.7],random_state=1); X=X@np.array([[1,0.4],[0,0.8]])
    fig,ax=plt.subplots(1,4,figsize=(11,2.9))
    for a,it in zip(ax,[1,3,8,60]):
        g=GaussianMixture(3,max_iter=it,init_params="random",random_state=3,tol=1e-12,covariance_type="full").fit(X)
        pts(a,X,GRY,8)
        for k in range(3):
            w,v=np.linalg.eigh(g.covariances_[k]); ang=np.degrees(np.arctan2(v[1,1],v[0,1]))
            for s in (1,2): a.add_patch(Ellipse(g.means_[k],2*s*np.sqrt(w[1]),2*s*np.sqrt(w[0]),angle=ang,fc="none",ec=CL[k],lw=2))
        a.set_title(f"Iteração {it}",fontsize=9.5); clean(a,True)
    fig.tight_layout(); save(fig,"f6_em")

def f6_suave():
    X,_=make_blobs(300,centers=[[0,0],[2.6,0.6],[1.2,3.2]],cluster_std=0.75,random_state=2)
    g=GaussianMixture(3,random_state=0,n_init=3).fit(X); P=g.predict_proba(X)
    base=np.array([matplotlib.colors.to_rgb(c) for c in (IND,AMB,GRN)]); cols=P@base
    fig,ax=plt.subplots(1,2,figsize=(8,3.5))
    pts(ax[0],X,[CL[i] for i in g.predict(X)],20); ax[0].set_title("Rígido: uma cor por ponto"); clean(ax[0],True)
    ax[1].scatter(X[:,0],X[:,1],c=cols,s=26,edgecolors="white",lw=0.3); ax[1].set_title("Suave: cor = mistura das probabilidades"); clean(ax[1],True)
    amb=np.argsort(P.max(1))[:3]
    for i in amb: ax[1].annotate(" / ".join(f"{p:.0%}" for p in P[i]),xy=X[i],xytext=(X[i][0]+1.1,X[i][1]-1.3),fontsize=7,arrowprops=dict(arrowstyle="-",color=INK,lw=0.7),bbox=dict(fc="white",ec=MID,boxstyle="round,pad=0.2"))
    fig.tight_layout(); save(fig,"f6_suave")

def f6_bic():
    X,_=make_blobs(600,centers=4,cluster_std=1.1,random_state=11); ks=range(1,9)
    b=[];a=[]
    for k in ks:
        g=GaussianMixture(k,n_init=3,random_state=0).fit(X); b.append(g.bic(X)); a.append(g.aic(X))
    fig,ax=plt.subplots(figsize=(6,3.5)); ax.plot(list(ks),b,"o-",c=IND,lw=2.2,label="BIC"); ax.plot(list(ks),a,"s--",c=AMB,lw=2,label="AIC")
    kb=int(np.argmin(b))+1; ax.scatter([kb],[min(b)],s=240,facecolors="none",edgecolors=RED,lw=2.5); ax.annotate(f"mínimo: K = {kb}",xy=(kb,min(b)),xytext=(kb+1.3,min(b)+(max(b)-min(b))*0.35),arrowprops=dict(arrowstyle="->",color=RED),color=RED,fontweight="bold")
    ax.set_xlabel("Número de componentes (K)"); ax.set_ylabel("Critério (menor é melhor)"); ax.legend(frameon=False); ax.grid(alpha=0.3); ax.set_title("Escolha de K por BIC/AIC")
    save(fig,"f6_bic")

def f6_grafo():
    rng=np.random.default_rng(3); A=rng.normal([0,0],0.6,(18,2)); B=rng.normal([3.2,0.3],0.6,(18,2)); X=np.r_[A,B]
    G=kneighbors_graph(X,4,include_self=False).toarray(); G=np.maximum(G,G.T)
    lab=np.r_[np.zeros(18),np.ones(18)]
    fig,ax=plt.subplots(figsize=(6.6,3.6))
    for i in range(len(X)):
        for j in range(i+1,len(X)):
            if G[i,j]:
                cut=lab[i]!=lab[j]; ax.plot([X[i,0],X[j,0]],[X[i,1],X[j,1]],c=RED if cut else GRY,lw=2.4 if cut else 0.9,ls="--" if cut else "-",zorder=1)
    ax.scatter(X[:,0],X[:,1],c=[IND if l==0 else GRN for l in lab],s=90,edgecolors="white",zorder=3)
    ax.axvline(1.65,c=RED,alpha=0.35,lw=6); ax.text(1.65,2.05,"corte de menor custo\n(poucas arestas)",ha="center",color=RED,fontweight="bold",fontsize=8.5)
    ax.set_ylim(-1.6,2.8); clean(ax,True); ax.set_title("Comunidades densamente conectadas, ligadas por poucas arestas")
    save(fig,"f6_grafo")

def f6_espectral_vs_kmeans():
    from sklearn.cluster import SpectralClustering
    X,_=make_circles(500,factor=0.4,noise=0.05,random_state=0)
    a=KMeans(2,n_init=10,random_state=0).fit_predict(X); b=SpectralClustering(2,affinity="nearest_neighbors",n_neighbors=10,random_state=0).fit_predict(X)
    fig,ax=plt.subplots(1,2,figsize=(7.4,3.4))
    pts(ax[0],X,[CL[i] for i in a],12); ax[0].set_title("K-means (erra)"); clean(ax[0],True)
    pts(ax[1],X,[CL[i] for i in b],12); ax[1].set_title("Espectral (acerta)"); clean(ax[1],True)
    save(fig,"f6_espectral_vs_kmeans")

if __name__=="__main__":
    for n,f in list(globals().items()):
        if n.startswith("f") and n[1].isdigit() and callable(f): f()
