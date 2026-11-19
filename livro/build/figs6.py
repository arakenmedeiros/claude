# Figuras do Capítulo 13 (Tópicos avançados de agrupamento)
from fc import *
from sklearn.datasets import make_moons, make_blobs, make_checkerboard
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralBiclustering, MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score
import networkx as nx
from networkx.algorithms.community import louvain_communities

def f13_consenso():
    X,y=make_moons(400,noise=0.08,random_state=0); rng=np.random.default_rng(0); P=[]
    for _ in range(40):
        k=int(rng.integers(8,16)); idx=rng.choice(len(X),320,replace=False)
        P.append(KMeans(k,n_init=1,random_state=int(rng.integers(10**6))).fit(X[idx]).predict(X))
    C=np.mean([p[:,None]==p[None,:] for p in P],axis=0)
    cons=AgglomerativeClustering(n_clusters=2,metric="precomputed",linkage="average").fit_predict(1-C)
    fig,ax=plt.subplots(1,4,figsize=(11,2.9))
    for a,p,t in ((ax[0],P[0],"Partição 1 (k=%d)"%(P[0].max()+1)),(ax[1],P[1],"Partição 2 (k=%d)"%(P[1].max()+1))):
        pts(a,X,[CL[i%8] for i in p],10); a.set_title(t); clean(a,True)
    ordem=np.argsort(cons*1000+np.argsort(np.argsort(X[:,0])))
    ax[2].imshow(C[ordem][:,ordem],cmap="Blues",aspect="auto"); ax[2].set_title("Coassociação (ordenada)"); clean(ax[2],True)
    pts(ax[3],X,[CL[i] for i in cons],10); ax[3].set_title("Consenso (ARI = %.2f)"%adjusted_rand_score(y,cons)); clean(ax[3],True)
    save(fig,"f13_consenso")

def f13_restricoes():
    X,y=make_blobs(120,centers=3,cluster_std=2.4,random_state=3); rng=np.random.default_rng(1)
    fig,ax=plt.subplots(figsize=(5.2,3.6)); pts(ax,X,GRY,22)
    from scipy.spatial.distance import cdist
    D=cdist(X,X); n=0; usados=set()
    for i in rng.permutation(len(X)):                    # must-link: vizinhos próximos do mesmo grupo verdadeiro
        cand=[j for j in np.argsort(D[i])[1:6] if y[j]==y[i] and j not in usados and i not in usados]
        if cand and n<8: ax.plot(*X[[i,cand[0]]].T,c=GRN,lw=2.2); usados|={i,cand[0]}; n+=1
    n=0
    for i in rng.permutation(len(X)):                    # cannot-link: vizinhos próximos de grupos verdadeiros diferentes
        cand=[j for j in np.argsort(D[i])[1:8] if y[j]!=y[i] and j not in usados and i not in usados]
        if cand and n<8: ax.plot(*X[[i,cand[0]]].T,c=RED,lw=2.2,ls="--"); usados|={i,cand[0]}; n+=1
    ax.plot([],[],c=GRN,lw=2,label="must-link (devem ficar juntos)"); ax.plot([],[],c=RED,lw=2,ls="--",label="cannot-link (devem ficar separados)")
    ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.02),ncol=1,frameon=False,fontsize=8); clean(ax,True)
    ax.set_title("Conhecimento do especialista em forma de pares"); save(fig,"f13_restricoes")

def f13_biclustering():
    dados,l,c=make_checkerboard(shape=(300,200),n_clusters=(4,3),noise=8,shuffle=False,random_state=0)
    rng=np.random.default_rng(0); emb=dados[rng.permutation(300)][:,rng.permutation(200)]
    bc=SpectralBiclustering(n_clusters=(4,3),method="log",random_state=0).fit(emb)
    ro=emb[np.argsort(bc.row_labels_)][:,np.argsort(bc.column_labels_)]
    fig,ax=plt.subplots(1,2,figsize=(7.4,3.4))
    ax[0].imshow(emb,cmap="viridis",aspect="auto"); ax[0].set_title("Como chega: usuários × itens"); ax[0].set_xlabel("itens"); ax[0].set_ylabel("usuários")
    ax[1].imshow(ro,cmap="viridis",aspect="auto"); ax[1].set_title("Após o biclustering: blocos"); ax[1].set_xlabel("itens (reordenados)")
    for a in ax: a.set_xticks([]); a.set_yticks([])
    save(fig,"f13_biclustering")

def f13_comunidades():
    G=nx.karate_club_graph(); pos=nx.spring_layout(G,seed=7)
    fig,ax=plt.subplots(1,2,figsize=(8,3.5))
    for a,res in zip(ax,(0.5,1.0)):
        com=louvain_communities(G,resolution=res,seed=0); rot={v:i for i,c in enumerate(com) for v in c}
        nx.draw_networkx_edges(G,pos,ax=a,edge_color=MID,width=1.0)
        nx.draw_networkx_nodes(G,pos,ax=a,node_color=[CL[rot[v]%8] for v in G],node_size=110,edgecolors="white",linewidths=0.8)
        a.set_title(f"Resolução {str(res).replace('.',',')}: {len(com)} comunidades"); a.axis("off")
    save(fig,"f13_comunidades")

def f13_fluxo():
    rng=np.random.default_rng(0); base=np.array([[0.,0.],[6.,0.],[3.,5.]])
    def lote(t,n=300):
        cen=base+np.array([[0.3*t,0.],[0.,0.25*t],[-0.2*t,0.]]); at=[0,1] if t<6 else [0,1,2]
        y=rng.choice(at,n); return cen[y]+rng.normal(0,0.8,(n,2)),y
    L=[lote(t) for t in range(12)]
    cong=KMeans(3,n_init=10,random_state=0).fit(L[0][0]); inc=MiniBatchKMeans(3,n_init=3,random_state=0); jan=[]
    a_c,a_i,a_j,ts=[],[],[],[]
    for t,(X,y) in enumerate(L):
        inc.partial_fit(X); jan.append(X); jan=jan[-3:]; mj=KMeans(3,n_init=5,random_state=0).fit(np.vstack(jan))
        if t>=1: ts.append(t); a_c.append(adjusted_rand_score(y,cong.predict(X))); a_i.append(adjusted_rand_score(y,inc.predict(X))); a_j.append(adjusted_rand_score(y,mj.predict(X)))
    fig,ax=plt.subplots(figsize=(6.6,3.3))
    ax.plot(ts,a_c,c=GRY,lw=2,marker="o",ms=4,label="modelo congelado"); ax.plot(ts,a_i,c=AMB,lw=2,marker="o",ms=4,label="incremental (partial_fit)"); ax.plot(ts,a_j,c=IND,lw=2.4,marker="o",ms=4,label="janela deslizante")
    ax.axvline(5.5,c=RED,ls="--",lw=1.2); ax.text(5.6,0.08,"novo grupo\napareceu",color=RED,fontsize=8.5)
    ax.set_xlabel("lote (tempo)"); ax.set_ylabel("ARI no lote"); ax.set_ylim(0,1.05); ax.legend(frameon=False,fontsize=8,loc="lower left",bbox_to_anchor=(0.0,0.2))
    ax.set_title("Deriva de conceito: quem acompanha o fluxo?"); save(fig,"f13_fluxo")

def f13_foraamostra():
    X,y=make_blobs(900,centers=4,cluster_std=1.0,random_state=7)
    from sklearn.model_selection import train_test_split
    Xtr,Xte=train_test_split(X,test_size=0.3,random_state=0); km=KMeans(4,n_init=10,random_state=0).fit(Xtr)
    lim=np.quantile(km.transform(Xtr).min(1),0.99); novos=np.array([[15,15],[-15,12],[16,13]]); rej=km.transform(novos).min(1)>lim
    fig,ax=plt.subplots(figsize=(5.4,3.8)); pts(ax,Xtr,[CL[i] for i in km.labels_],12,alpha=0.5)
    for c in km.cluster_centers_: ax.add_patch(Circle(c,lim,fc="none",ec=INK,lw=1,ls="--"))
    ax.scatter(*Xte[:40].T,marker="x",c=INK,s=22,label="casos novos aceitos"); ax.scatter(*novos.T,marker="*",c=RED,s=140,label="rejeitados (fora do raio)")
    ax.legend(frameon=False,fontsize=8,loc="lower left"); ax.set_title("Atribuir ou rejeitar casos novos"); clean(ax,True); save(fig,"f13_foraamostra")

if __name__=="__main__":
    f13_consenso(); f13_restricoes(); f13_biclustering(); f13_comunidades(); f13_fluxo(); f13_foraamostra()
