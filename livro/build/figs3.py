from fc import *
from sklearn.datasets import make_blobs, load_iris, load_digits, make_swiss_roll
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import MDS, TSNE, Isomap
from sklearn.neighbors import NearestNeighbors

def f7_familias():
    fig,ax=canvas(9,3.6,(0,100),(0,40))
    cards=[("Interna","usa só os dados e os grupos","mede coesão e separação\n(silhueta, Calinski-Harabasz,\nDavies-Bouldin)",IND),("Externa","compara com rótulos de referência","ARI, NMI, pureza,\nhomogeneidade, completude",GRN),("Relativa","compara várias soluções","varia k ou parâmetros\ne escolhe a melhor\n(gap, estabilidade, cotovelo)",AMB)]
    for i,(t,s1,s2,c) in enumerate(cards):
        x=1+i*33
        rbox(ax,x,1,31,37,fc="white",ec=c,lw=2.2,r=3); rbox(ax,x,29,31,9,t,fc=c,tc="white",bold=True,fs=11,r=3)
        ax.text(x+15.5,23.5,s1.replace(" de referência","\nde referência") if "rótulos" in s1 else s1,ha="center",va="center",fontsize=7.8,color=c,fontweight="bold"); ax.text(x+15.5,12.5,s2,ha="center",va="center",fontsize=8.3)
    save(fig,"f7_familias")

def f7_tendencia():
    rng=np.random.default_rng(0); U=rng.random((300,2)); C,_=make_blobs(300,centers=4,cluster_std=0.6,random_state=2)
    def hop(X,seed=42):
        r=np.random.default_rng(seed); n,d=X.shape; m=30; nn=NearestNeighbors(n_neighbors=2).fit(X); idx=r.choice(n,m,replace=False)
        w=nn.kneighbors(X[idx],2)[0][:,1]; Uu=r.uniform(X.min(0),X.max(0),(m,d)); u=nn.kneighbors(Uu,1)[0][:,0]; return (u**d).sum()/((u**d).sum()+(w**d).sum())
    fig,ax=plt.subplots(1,2,figsize=(8,3.5))
    pts(ax[0],U,GRY,16); ax[0].set_title(f"Uniforme: Hopkins ≈ {hop(U):.2f}\\n(sem grupos)".replace("\\n","\n")); clean(ax[0],True)
    pts(ax[1],C,[CL[i] for i in KMeans(4,n_init=5,random_state=0).fit_predict(C)],16); ax[1].set_title(f"Agrupado: Hopkins ≈ {hop(C):.2f}\n(forte tendência)"); clean(ax[1],True)
    save(fig,"f7_tendencia")

def f7_silhueta():
    X,_=make_blobs(400,centers=4,random_state=8)
    fig,ax=plt.subplots(1,2,figsize=(9,3.8))
    for a,k in zip(ax,[4,6]):
        l=KMeans(k,n_init=10,random_state=0).fit_predict(X); s=silhouette_samples(X,l); y0=10
        for c in range(k):
            v=np.sort(s[l==c]); a.fill_betweenx(np.arange(y0,y0+len(v)),0,v,color=CL[c%8],alpha=0.85); a.text(-0.08,y0+len(v)/2,str(c),fontsize=8); y0+=len(v)+10
        a.axvline(s.mean(),c=RED,ls="--",lw=1.6); a.set_title(f"k = {k}  |  silhueta média = {s.mean():.2f}"); a.set_xlabel("Coeficiente de silhueta"); a.set_yticks([]); a.set_xlim(-0.2,1)
    fig.tight_layout(); save(fig,"f7_silhueta")

def f7_indices():
    X,_=make_blobs(500,centers=4,random_state=8); ks=range(2,9); S=[];C=[];D=[]
    for k in ks:
        l=KMeans(k,n_init=10,random_state=0).fit_predict(X); S.append(silhouette_score(X,l)); C.append(calinski_harabasz_score(X,l)); D.append(davies_bouldin_score(X,l))
    fig,ax=plt.subplots(1,3,figsize=(10.6,3.2))
    for a,v,t,c,best in [(ax[0],S,"Silhueta (maior = melhor)",IND,np.argmax(S)),(ax[1],C,"Calinski-Harabasz (maior = melhor)",GRN,np.argmax(C)),(ax[2],D,"Davies-Bouldin (menor = melhor)",AMB,np.argmin(D))]:
        a.plot(list(ks),v,"o-",c=c,lw=2.2); a.scatter([list(ks)[best]],[v[best]],s=200,facecolors="none",edgecolors=RED,lw=2.4); a.set_title(t,fontsize=9); a.set_xlabel("k"); a.grid(alpha=0.3)
    fig.tight_layout(); save(fig,"f7_indices")

def f7_gap():
    X,_=make_blobs(300,centers=4,random_state=3); rng=np.random.default_rng(0); ks=list(range(1,9)); B=10
    W=lambda Z,k: np.log(KMeans(k,n_init=5,random_state=0).fit(Z).inertia_)
    obs=[W(X,k) for k in ks]; ref=np.array([[W(rng.uniform(X.min(0),X.max(0),X.shape),k) for k in ks] for _ in range(B)])
    gap=ref.mean(0)-obs; sd=ref.std(0)*np.sqrt(1+1/B)
    fig,ax=plt.subplots(figsize=(6,3.5)); ax.errorbar(ks,gap,yerr=sd,fmt="o-",c=IND,lw=2.2,capsize=3)
    kb=next((k for i,k in enumerate(ks[:-1]) if gap[i]>=gap[i+1]-sd[i+1]),ks[-1]); ax.scatter([kb],[gap[kb-1]],s=240,facecolors="none",edgecolors=RED,lw=2.4)
    ax.annotate(f"escolha: k = {kb}",xy=(kb,gap[kb-1]),xytext=(kb+1.2,gap[kb-1]-0.25),arrowprops=dict(arrowstyle="->",color=RED),color=RED,fontweight="bold")
    ax.set_xlabel("k"); ax.set_ylabel("Gap(k)"); ax.set_title("Estatística gap"); ax.grid(alpha=0.3); save(fig,"f7_gap")
    print("gap k",kb)

def f7_estabilidade():
    X,_=make_blobs(600,centers=4,random_state=8); rng=np.random.default_rng(0); ks=range(2,8); data=[]
    for k in ks:
        sc=[]
        for _ in range(15):
            i1=rng.choice(len(X),300,replace=False); i2=rng.choice(len(X),300,replace=False)
            m1=KMeans(k,n_init=3,random_state=0).fit(X[i1]); m2=KMeans(k,n_init=3,random_state=1).fit(X[i2]); sc.append(adjusted_rand_score(m1.predict(X),m2.predict(X)))
        data.append(sc)
    fig,ax=plt.subplots(figsize=(6.2,3.5)); bp=ax.boxplot(data,tick_labels=list(ks),patch_artist=True,widths=0.55)
    for p in bp["boxes"]: p.set(facecolor="#C7D2FE",edgecolor=IND)
    for m in bp["medians"]: m.set(color=RED,lw=2)
    ax.set_xlabel("k"); ax.set_ylabel("Concordância entre repetições (ARI)"); ax.set_title("Estabilidade: alta concordância sugere grupos reais"); ax.grid(alpha=0.3,axis="y")
    save(fig,"f7_estabilidade")

def f7_contingencia():
    M=np.array([[48,2,0],[1,45,6],[1,3,44]]); fig,ax=plt.subplots(figsize=(4.6,3.8)); im=ax.imshow(M,cmap="Blues")
    ax.set_xticks(range(3)); ax.set_xticklabels(["Classe A","Classe B","Classe C"]); ax.set_yticks(range(3)); ax.set_yticklabels(["Grupo 0","Grupo 1","Grupo 2"])
    for i in range(3):
        for j in range(3): ax.text(j,i,M[i,j],ha="center",va="center",color="white" if M[i,j]>25 else INK,fontsize=12,fontweight="bold")
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title("Diagonal forte = boa concordância"); save(fig,"f7_contingencia")

def f8_sombra():
    rng=np.random.default_rng(2); P=rng.normal(size=(300,3))*np.array([3.0,1.6,0.35]); th=0.7
    R=np.array([[np.cos(th),-np.sin(th),0],[np.sin(th),np.cos(th),0],[0,0,1]]); R2=np.array([[1,0,0],[0,np.cos(0.5),-np.sin(0.5)],[0,np.sin(0.5),np.cos(0.5)]]); P=P@R.T@R2.T
    fig=plt.figure(figsize=(10.4,3.2)); a=fig.add_subplot(1,4,1,projection="3d"); a.scatter(*P.T,c=IND,s=6,alpha=0.7); a.set_title("Objeto 3D",fontsize=9.5); a.set_xticklabels([]); a.set_yticklabels([]); a.set_zticklabels([])
    pca=PCA(2).fit(P); u,s,vt=np.linalg.svd(P-P.mean(0),full_matrices=False)
    Q,_=np.linalg.qr(rng.normal(size=(3,2))); worst=vt[1:3].T
    projs=[("Ângulo ruim\n(perde quase tudo)",(P-P.mean(0))@worst,RED),("Ângulo médio",(P-P.mean(0))@Q,AMB),("Melhor ângulo = PCA\n(preserva mais variância)",(P-P.mean(0))@vt[:2].T,GRN)]
    for i,(t,Z,c) in enumerate(projs):
        b=fig.add_subplot(1,4,i+2); b.scatter(Z[:,0],Z[:,1],c=c,s=7,alpha=0.7); b.set_xlim(-8,8); b.set_ylim(-8,8); b.set_aspect("equal"); b.set_title(t,fontsize=9.5); clean(b,True)
        b.text(0.5,0.04,f"variância: {Z.var(0).sum():.1f}",transform=b.transAxes,ha="center",fontsize=8)
    save(fig,"f8_sombra",tight=False)

def f8_pca_intuicao():
    rng=np.random.default_rng(1); X=rng.multivariate_normal([0,0],[[3.2,2.2],[2.2,2.0]],200)
    p=PCA(2).fit(X); fig,ax=plt.subplots(1,2,figsize=(8.8,3.8))
    pts(ax[0],X,IND,16)
    for k,(c,t) in enumerate([(RED,"PC1"),(GRN,"PC2")]):
        v=p.components_[k]*np.sqrt(p.explained_variance_[k])*2.2; ax[0].annotate("",xy=v,xytext=(0,0),arrowprops=dict(arrowstyle="-|>",color=c,lw=3)); ax[0].text(v[0]*1.1,v[1]*1.1,t,color=c,fontsize=11,fontweight="bold")
    ax[0].set_aspect("equal"); ax[0].set_title("Eixos originais e componentes principais"); ax[0].axhline(0,c=MID); ax[0].axvline(0,c=MID); clean(ax[0],True)
    Z=p.transform(X); pts(ax[1],Z,IND,16); ax[1].axhline(0,c=GRN,lw=1.5); ax[1].axvline(0,c=RED,lw=1.5); ax[1].set_aspect("equal"); ax[1].set_title("Dados 'girados': PC1 horizontal")
    ax[1].text(0.97,0.06,f"PC1 explica {p.explained_variance_ratio_[0]:.0%}",transform=ax[1].transAxes,ha="right",color=RED,fontweight="bold"); clean(ax[1],True)
    fig.tight_layout(); save(fig,"f8_pca_intuicao")

def f8_scree():
    X=StandardScaler().fit_transform(load_digits().data); X=X[:,X.std(0)>0]
    p=PCA().fit(X); ev=p.explained_variance_ratio_; cum=ev.cumsum()
    fig,ax=plt.subplots(1,2,figsize=(9.2,3.4)); ax[0].plot(range(1,31),ev[:30],"o-",c=IND,lw=2); ax[0].set_title("Gráfico scree"); ax[0].set_xlabel("Componente"); ax[0].set_ylabel("Variância explicada"); ax[0].grid(alpha=0.3)
    ax[1].plot(range(1,len(cum)+1),cum,c=GRN,lw=2.4); ax[1].axhline(0.95,c=RED,ls="--"); k95=int(np.argmax(cum>=0.95))+1; ax[1].axvline(k95,c=RED,ls="--"); ax[1].text(k95+1,0.5,f"{k95} componentes\n→ 95 %",color=RED,fontweight="bold")
    ax[1].set_title("Variância acumulada"); ax[1].set_xlabel("Nº de componentes"); ax[1].grid(alpha=0.3)
    fig.tight_layout(); save(fig,"f8_scree")

def f8_biplot():
    d=load_iris(); X=StandardScaler().fit_transform(d.data); p=PCA(2).fit(X); Z=p.transform(X)
    fig,ax=plt.subplots(figsize=(5.8,4.4))
    for c,n in enumerate(d.target_names): ax.scatter(Z[d.target==c,0],Z[d.target==c,1],c=CL[c],s=22,label=n,edgecolors="white",lw=0.3)
    nm=["sépala compr.","sépala larg.","pétala compr.","pétala larg."]
    for j in range(4):
        v=p.components_[:,j]*3; ax.annotate("",xy=v,xytext=(0,0),arrowprops=dict(arrowstyle="-|>",color=INK,lw=1.8)); ax.text(v[0]*1.1,v[1]*1.1,nm[j],fontsize=8,fontweight="bold",ha="center")
    ax.set_xlabel(f"PC1 ({p.explained_variance_ratio_[0]:.0%})"); ax.set_ylabel(f"PC2 ({p.explained_variance_ratio_[1]:.0%})"); ax.legend(fontsize=8,frameon=False,loc="lower right"); ax.axhline(0,c=MID,lw=0.8); ax.axvline(0,c=MID,lw=0.8)
    save(fig,"f8_biplot")

def f8_digitos():
    X,y=load_digits(return_X_y=True); idx=[0,1,2,3,4,5,6,7]
    ks=[2,5,15,40]; fig,ax=plt.subplots(5,8,figsize=(8,5.3))
    for j,i in enumerate(idx): ax[0,j].imshow(X[i].reshape(8,8),cmap="gray_r")
    for r,k in enumerate(ks,1):
        p=PCA(k).fit(X); R=p.inverse_transform(p.transform(X))
        for j,i in enumerate(idx): ax[r,j].imshow(R[i].reshape(8,8),cmap="gray_r")
    for a in ax.ravel(): clean(a)
    ax[0,0].set_ylabel("original",rotation=0,ha="right",va="center",fontsize=8.5)
    for r,k in enumerate(ks,1): ax[r,0].set_ylabel(f"{k} comp.",rotation=0,ha="right",va="center",fontsize=8.5)
    fig.tight_layout(); save(fig,"f8_digitos")

def f8_ica():
    t=np.linspace(0,8,1500); S=np.c_[np.sin(2*t),np.sign(np.sin(3*t))]+0.04*np.random.default_rng(0).normal(size=(1500,2))
    A=np.array([[1,0.5],[0.4,1]]); X=S@A.T; Si=FastICA(2,random_state=0).fit_transform(X); Sp=PCA(2).fit_transform(X)
    fig,ax=plt.subplots(4,2,figsize=(8,5.6),sharex=True)
    rows=[("Fontes originais",S,IND),("Microfones (mistura)",X,AMB),("ICA recupera",Si,GRN),("PCA falha em separar",Sp,RED)]
    for r,(t_,M,c) in enumerate(rows):
        for j in range(2): ax[r,j].plot(t,M[:,j],c=c,lw=1.1); ax[r,j].set_yticks([]); ax[r,j].spines["left"].set_visible(False)
        ax[r,0].set_ylabel(t_,fontsize=8.5,rotation=0,ha="right",va="center")
    ax[0,0].set_title("sinal 1"); ax[0,1].set_title("sinal 2"); fig.tight_layout(); save(fig,"f8_ica")

def f8_nmf():
    X,_=load_digits(return_X_y=True); X=X/16
    n=NMF(8,init="nndsvda",random_state=0,max_iter=600).fit(X); p=PCA(8).fit(X)
    fig,ax=plt.subplots(2,8,figsize=(8.6,2.8))
    for j in range(8):
        ax[0,j].imshow(n.components_[j].reshape(8,8),cmap="Blues"); v=p.components_[j].reshape(8,8); m=abs(v).max(); ax[1,j].imshow(v,cmap="RdBu_r",vmin=-m,vmax=m)
    for a in ax.ravel(): clean(a)
    ax[0,0].set_ylabel("NMF\n(partes ≥ 0)",rotation=0,ha="right",va="center",fontsize=9); ax[1,0].set_ylabel("PCA\n(+ e −)",rotation=0,ha="right",va="center",fontsize=9)
    fig.tight_layout(); save(fig,"f8_nmf")

def f9_rocambole():
    X,t=make_swiss_roll(1500,random_state=0); fig=plt.figure(figsize=(9.2,3.9))
    a=fig.add_subplot(1,2,1,projection="3d"); a.scatter(X[:,0],X[:,1],X[:,2],c=t,cmap="viridis",s=6); a.set_title("Em 3D: enrolado"); a.view_init(10,-70)
    b=fig.add_subplot(1,2,2); b.scatter(t,X[:,1],c=t,cmap="viridis",s=6); b.set_title("Desenrolado em 2D: estrutura real"); b.set_xlabel("posição ao longo do rocambole"); b.set_ylabel("largura")
    for ax in (a,): ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    save(fig,"f9_rocambole",tight=False)

def f9_mds_cidades():
    C={"São Paulo":(-23.55,-46.63),"Rio de Janeiro":(-22.91,-43.17),"Brasília":(-15.78,-47.93),"Salvador":(-12.97,-38.51),"Fortaleza":(-3.73,-38.53),"Manaus":(-3.12,-60.02),"Porto Alegre":(-30.03,-51.23),"Belo Horizonte":(-19.92,-43.94),"Recife":(-8.05,-34.88),"Curitiba":(-25.43,-49.27)}
    names=list(C); ll=np.radians(np.array([C[n] for n in names])); R=6371
    def hav(a,b):
        dl=b[0]-a[0]; dn=b[1]-a[1]; h=np.sin(dl/2)**2+np.cos(a[0])*np.cos(b[0])*np.sin(dn/2)**2; return 2*R*np.arcsin(np.sqrt(h))
    D=np.array([[hav(a,b) for b in ll] for a in ll])
    m=MDS(2,dissimilarity="precomputed",random_state=0,n_init=8,normalized_stress="auto"); Y=m.fit_transform(D)
    from scipy.spatial import procrustes
    true=np.c_[np.array([C[n][1] for n in names]),np.array([C[n][0] for n in names])]
    _,Yp,_=procrustes(true,Y)
    fig,ax=plt.subplots(1,2,figsize=(10,4.2),gridspec_kw={"width_ratios":[1,1.2]})
    im=ax[0].imshow(D,cmap="YlGnBu"); ax[0].set_xticks(range(10)); ax[0].set_yticks(range(10)); ax[0].set_xticklabels(names,rotation=70,fontsize=7); ax[0].set_yticklabels(names,fontsize=7); ax[0].set_title("Entrada: só a tabela de distâncias (km)")
    tz=(true-true.mean(0))/np.linalg.norm(true-true.mean(0))
    ax[1].scatter(Yp[:,0],Yp[:,1],c=IND,s=70,zorder=3,edgecolors="white")
    for n,(x,y) in zip(names,Yp): ax[1].text(x+0.01,y+0.012,n,fontsize=8)
    ax[1].set_title("Saída: mapa reconstruído pelo MDS"); clean(ax[1],True); ax[1].set_aspect("equal")
    fig.tight_layout(); save(fig,"f9_mds_cidades")

def f9_geodesica():
    th=np.linspace(0.6*np.pi,3.6*np.pi,220); r=th; x=r*np.cos(th); y=r*np.sin(th); rng=np.random.default_rng(0); X=np.c_[x,y]+rng.normal(0,0.12,(220,2))
    i,j=15,150; fig,ax=plt.subplots(figsize=(5.4,5))
    ax.scatter(X[:,0],X[:,1],c=GRY,s=14); 
    nn=NearestNeighbors(n_neighbors=3).fit(X); _,nb=nn.kneighbors(X)
    for a in range(len(X)):
        for b in nb[a,1:]: ax.plot([X[a,0],X[b,0]],[X[a,1],X[b,1]],c=MID,lw=0.7,zorder=0)
    ax.plot([X[i,0],X[j,0]],[X[i,1],X[j,1]],c=RED,ls="--",lw=2.4,label="Euclidiana (linha reta)")
    ax.plot(x[i:j+1],y[i:j+1],c=IND,lw=3.2,label="Geodésica (ao longo da variedade)")
    ax.scatter(*X[[i,j]].T,c=[AMB,AMB],s=110,zorder=5,edgecolors="white"); ax.set_aspect("equal"); ax.legend(fontsize=8,loc="lower left"); clean(ax,True); ax.set_title("Dois pontos 'vizinhos' em linha reta,\nmas distantes pelo caminho real")
    save(fig,"f9_geodesica")

def f9_tsne_idea():
    X,y=load_digits(return_X_y=True); idx=np.arange(0,1797,3); Z=TSNE(2,perplexity=30,init="pca",random_state=0).fit_transform(X[idx]); yy=y[idx]
    rng=np.random.default_rng(0); R=rng.normal(size=Z.shape)*Z.std()*0.8; M=0.5*R+0.5*Z+rng.normal(size=Z.shape)*2
    fig,ax=plt.subplots(1,3,figsize=(10.4,3.5))
    for a,(P,t) in zip(ax,[(R,"Início: posições aleatórias"),(M,"Meio: vizinhos se aproximam"),(Z,"Fim: rodinhas formadas")]):
        a.scatter(P[:,0],P[:,1],c=yy,cmap="tab10",s=9); a.set_title(t,fontsize=9.5); clean(a,True)
    fig.text(0.5,-0.04,"Ilustração esquemática (painéis inicial e intermediário são ilustrativos): atração entre vizinhos e repulsão entre não vizinhos.",ha="center",fontsize=7.5,style="italic",color=GRY)
    save(fig,"f9_tsne_idea")

def f9_tsne_perplexidade():
    X,y=load_digits(return_X_y=True); fig,ax=plt.subplots(1,3,figsize=(10.6,3.4))
    for a,p in zip(ax,[5,30,100]):
        Z=TSNE(2,perplexity=p,init="pca",learning_rate="auto",random_state=0).fit_transform(X); a.scatter(Z[:,0],Z[:,1],c=y,cmap="tab10",s=5); a.set_title(f"Perplexidade = {p}",fontsize=10); clean(a,True)
    fig.tight_layout(); save(fig,"f9_tsne_perplexidade")

def f9_comparacao():
    import umap
    X,y=load_digits(return_X_y=True); fig,ax=plt.subplots(1,4,figsize=(12,3.2))
    for a,(n,Z) in zip(ax,[("PCA",PCA(2).fit_transform(X)),("Isomap",Isomap(n_neighbors=10).fit_transform(X)),("t-SNE",TSNE(2,init="pca",random_state=0).fit_transform(X)),("UMAP",umap.UMAP(random_state=0).fit_transform(X))]):
        a.scatter(Z[:,0],Z[:,1],c=y,cmap="tab10",s=4); a.set_title(n,fontsize=10.5); clean(a,True)
    fig.tight_layout(); save(fig,"f9_comparacao")

if __name__=="__main__":
    import sys
    for n,f in list(globals().items()):
        if n.startswith("f") and n[1].isdigit() and callable(f): f()
