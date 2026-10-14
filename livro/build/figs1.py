from fc import *
from sklearn.datasets import make_blobs, make_moons, make_circles, load_iris
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi

def f1_festa():
    X,_=make_blobs(150,centers=[[0,0],[4,1],[2,4.5]],cluster_std=0.6,random_state=3)
    fig,ax=plt.subplots(1,2,figsize=(8,3.5))
    pts(ax[0],X,GRY,28); ax[0].set_title("Antes: convidados sem etiqueta"); clean(ax[0],True)
    lab=KMeans(3,n_init=10,random_state=0).fit_predict(X)
    pts(ax[1],X,[CL[i] for i in lab],28); ax[1].set_title("Depois: grupos descobertos"); clean(ax[1],True)
    for k in range(3):
        c=X[lab==k].mean(0); ax[1].add_patch(Circle(c,1.3,fc="none",ec=CL[k],lw=1.6,ls="--"))
    fig.text(0.5,0.02,"nenhuma resposta foi fornecida ao algoritmo",ha="center",fontsize=8.5,style="italic",color=GRY)
    save(fig,"f1_festa")

def f1_tipos():
    fig,ax=canvas(8.4,5.2,(0,100),(0,64))
    cards=[("Supervisionado","dados + rótulos (X e y)","feedback: gabarito",IND,0,33),
           ("Não supervisionado","apenas dados (X)","feedback: nenhum — descobre estrutura",GRN,51,33),
           ("Auto-supervisionado","dados; rótulo criado dos próprios dados","feedback: tarefa-pretexto",VIO,0,0),
           ("Por reforço","agente age em um ambiente","feedback: recompensa / punição",AMB,51,0)]
    rng=np.random.default_rng(1)
    for i,(t,s1,s2,c,x0,y0) in enumerate(cards):
        rbox(ax,x0+1,y0+1,47,30,fc="white",ec=c,lw=2,r=2.5)
        rbox(ax,x0+1,y0+24,47,7,t,fc=c,tc="white",bold=True,fs=10.5,r=2.5)
        ax.text(x0+3,y0+20,s1,fontsize=8.2,va="center"); ax.text(x0+3,y0+16.5,s2,fontsize=8.2,va="center",color=c,fontweight="bold")
        if i==0:
            for k,(cx,cy) in enumerate([(12,8),(27,9),(20,13)]):
                P=rng.normal([cx,cy],2.0,(10,2)); ax.scatter(P[:,0]+x0,P[:,1]+y0,c=CL[[0,1,2][k]],s=14,edgecolors="white",lw=0.3)
            ax.text(x0+35,y0+9,"y = gato\ny = cão\ny = pássaro",fontsize=7.5,va="center")
        elif i==1:
            for k,(cx,cy) in enumerate([(12,8),(27,9),(20,13)]):
                P=rng.normal([cx,cy],2.0,(10,2)); ax.scatter(P[:,0]+x0,P[:,1]+y0,c=GRY,s=14,edgecolors="white",lw=0.3)
            ax.text(x0+40,y0+9,"?",fontsize=26,color=GRN,ha="center",va="center",fontweight="bold")
        elif i==2:
            rbox(ax,x0+4,y0+5,16,6,"A casa é ___",fc=LIGHT,fs=8); arrow(ax,(x0+21,y0+8),(x0+27,y0+8),VIO); rbox(ax,x0+28,y0+5,17,6,"prever 'azul'",fc="#EDE9FE",fs=8,ec=VIO)
        else:
            rbox(ax,x0+4,y0+4,14,8,"Agente",fc="#FEF3C7",ec=AMB,fs=8); rbox(ax,x0+31,y0+4,14,8,"Ambiente",fc=LIGHT,ec=GRY,fs=8)
            arrow(ax,(x0+18,y0+10),(x0+31,y0+10),AMB,rad=-0.3); arrow(ax,(x0+31,y0+6),(x0+18,y0+6),GRN,rad=-0.3)
            ax.text(x0+24.5,y0+13.8,"ação",fontsize=7.5,ha="center"); ax.text(x0+24.5,y0+1.8,"recompensa",fontsize=7.5,ha="center",color=GRN)
    save(fig,"f1_tipos")

def f1_tarefas():
    fig,ax=canvas(8.4,5.6,(0,100),(0,70))
    rbox(ax,33,27,34,16,"APRENDIZADO\nNÃO SUPERVISIONADO\n(só X, sem y)",fc=IND,tc="white",bold=True,fs=10.5,r=4)
    cfg=[("1. Agrupamento",CYA,2,40),("2. Redução de\ndimensionalidade",GRN,2,2),("3. Regras de\nassociação",AMB,70,40),("4. Detecção de\nanomalias",RED,70,2)]
    rng=np.random.default_rng(0)
    for (t,c,x0,y0) in cfg:
        rbox(ax,x0,y0,28,26,fc="white",ec=c,lw=2.2,r=3); rbox(ax,x0,y0+17.5,28,8.5,t,fc=c,tc="white",bold=True,fs=8.8,r=3)
    # mini ilustracoes
    for k,(cx,cy) in enumerate([(9,46),(21,45),(15,53)]):
        P=rng.normal([cx,cy],1.3,(8,2)); ax.scatter(P[:,0],P[:,1],c=CL[[4,1,2][k]],s=14)
    # reducao: 3D->2D
    ax.add_patch(Polygon([[5,5],[14,8],[14,16],[5,13]],fc="#D1FAE5",ec=GRN)); arrow(ax,(16,11),(21,11),GRN)
    ax.plot([22,27],[11,11],c=GRN,lw=3)
    # associacao
    for i,(t,c) in enumerate([("pão",AMB),("leite",CYA),("café",INK)]):
        rbox(ax,72+i*8.6,47,8,4.5,t,fc=MID,fs=6.8)
    ax.plot([76,84],[46.6,46.6],c=AMB,lw=1.5); ax.text(84,44.2,"juntos!",fontsize=7.5,ha="center",color=AMB,fontweight="bold")
    # anomalia
    P=rng.normal([82,9],1.6,(25,2)); ax.scatter(P[:,0],P[:,1],c=GRY,s=10); ax.scatter([94],[15],c=RED,s=70,marker="X")
    for (c,x0,y0,dx,dy) in [(CYA,30,48,3,0),(GRN,30,14,3,0),(AMB,70,48,-3,0),(RED,70,14,-3,0)]:
        pass
    arrow(ax,(32,43),(26,52),CYA); arrow(ax,(33,29),(26,20),GRN); arrow(ax,(67,43),(72,52),AMB); arrow(ax,(67,29),(72,20),RED)
    ax.text(15,36.5,"Quem se parece\ncom quem?",fontsize=7.8,ha="center",color=CYA,fontweight="bold",va="center")
    ax.text(15,-1.5,"Como resumir sem\nperder o essencial?",fontsize=7.8,ha="center",color=GRN,fontweight="bold",va="top")
    ax.text(84,36.5,"O que aparece\njunto?",fontsize=7.8,ha="center",color=AMB,fontweight="bold",va="center")
    ax.text(84,-1.5,"Quem foge\ndo normal?",fontsize=7.8,ha="center",color=RED,fontweight="bold",va="top")
    save(fig,"f1_tarefas")

def f1_historia():
    ev=[(1901,"Pearson\n(precursor do PCA)",IND),(1933,"Hotelling\nformaliza o PCA",IND),(1957,"Lloyd:\nK-means*",CYA),(1977,"Algoritmo EM",GRN),(1993,"Regras de\nassociação",AMB),
        (1996,"DBSCAN",CYA),(2000,"Isomap\ne LLE",VIO),(2006,"Autoencoders\nprofundos",PNK),(2008,"t-SNE",VIO),(2014,"VAE e GAN",PNK),(2018,"UMAP",VIO),(2020,"Aprendizado\ncontrastivo",RED)]
    fig,ax=plt.subplots(figsize=(10,3.8)); ax.set_xlim(-0.8,11.8); ax.set_ylim(-5,5); ax.axis("off")
    ax.plot([-0.5,11.5],[0,0],c=INK,lw=2.5)
    for i,(y,t,c) in enumerate(ev):
        up=(i%2==0); yy=2.4 if up else -2.4
        ax.plot([i,i],[0,yy*0.62],c=c,lw=1.5); ax.scatter([i],[0],s=70,c=c,zorder=3,edgecolors="white",lw=1.5)
        ax.text(i,yy*0.7+(0.4 if up else -0.4),f"{y}",ha="center",va="bottom" if up else "top",fontsize=9,fontweight="bold",color=c)
        ax.text(i,yy*0.7+(1.2 if up else -1.2),t,ha="center",va="bottom" if up else "top",fontsize=7.4)
    ax.text(-0.7,-4.8,"* proposto em 1957, publicado em 1982 (escala do eixo não proporcional ao tempo)",fontsize=7,style="italic",color=GRY)
    save(fig,"f1_historia")

def f1_pipeline():
    steps=["1. Entender\no problema","2. Coletar\nos dados","3. Explorar\ne limpar","4. Preparar\natributos","5. Escolher e\nrodar algoritmo","6. Avaliar\nresultados","7. Interpretar\ne comunicar","8. Agir e\nrepetir"]
    fig,ax=canvas(9,4.2,(0,100),(0,50))
    pos=[(3,28),(27,28),(51,28),(75,28),(75,4),(51,4),(27,4),(3,4)]
    for i,((x,y),t) in enumerate(zip(pos,steps)):
        rbox(ax,x,y,21,15,t,fc=CL[i%len(CL)],tc="white",bold=True,fs=8.6,r=3)
    for i in range(7):
        (x1,y1),(x2,y2)=pos[i],pos[i+1]
        if i<3: arrow(ax,(x1+21.3,y1+7.5),(x2-0.3,y2+7.5),INK)
        elif i==3: arrow(ax,(x1+10.5,y1-0.3),(x2+10.5,y2+15.3),INK)
        else: arrow(ax,(x1-0.3,y1+7.5),(x2+21.3,y2+7.5),INK)
    arrow(ax,(3+10.5,19.3),(3+10.5,27.7),GRY,rad=0,ls="--"); ax.text(1,23.5,"recomeça",rotation=90,fontsize=7.5,color=GRY,va="center")
    save(fig,"f1_pipeline")

def f1_roteiro():
    fig,ax=canvas(10,4.4,(0,100),(0,54))
    parts=[("Parte I\nFundamentos","Cap. 1 e 2\nConceitos,\ndados e\ndistâncias",IND),("Parte II\nAgrupamento","Cap. 3 a 7\nK-means,\nhierárquico,\ndensidade, GMM,\nespectral,\navaliação",CYA),
           ("Parte III\nRedução de\ndimensões","Cap. 8 e 9\nPCA, ICA, NMF,\nIsomap, t-SNE,\nUMAP",GRN),("Parte IV\nPadrões e\nanomalias","Cap. 10 e 11\nRegras de\nassociação,\nIsolation Forest,\nLOF, SVM",AMB),
           ("Parte V\nAprendizado\nprofundo","Cap. 12\nAutoencoders,\nVAEs, GANs,\ncontrastivo",RED)]
    for i,(t,d,c) in enumerate(parts):
        x=1+i*19.7; y=1+i*2.5
        rbox(ax,x,y+27,18,13,t,fc=c,tc="white",bold=True,fs=8,r=3)
        rbox(ax,x,y,18,26,d,fc="white",ec=c,lw=1.8,fs=7.6,r=3)
        if i<4: arrow(ax,(x+18.2,y+33),(x+19.5,y+35),INK)
    save(fig,"f1_roteiro")

def f2_matriz():
    fig,ax=canvas(9,3.6,(0,100),(0,44))
    cols=["Idade","Gasto R$","Visitas"]; rows=[("Ana",25,120,8),("Bruno",62,340,12),("Carla",31,95,4),("Diego",58,310,10)]
    x0,y0=4,8
    for j,c in enumerate(cols): rbox(ax,x0+16+j*12,y0+27,11.2,6,c,fc=IND,tc="white",bold=True,fs=7.8,r=1)
    ax.text(x0+16+17,y0+36,"atributos (colunas)  →",ha="center",fontsize=8.5,color=IND,fontweight="bold")
    for i,r in enumerate(rows):
        rbox(ax,x0,y0+20-i*6.5,15,6,r[0],fc=CYA,tc="white",bold=True,fs=8,r=1)
        for j in range(3): rbox(ax,x0+16+j*12,y0+20-i*6.5,11.2,6,str(r[j+1]),fc=LIGHT if i!=1 else "#FEF3C7",fs=8.5,r=1,ec=AMB if i==1 else None)
    ax.text(x0+7,y0+5,"exemplos\n(linhas)",ha="center",fontsize=8.5,color=CYA,fontweight="bold",va="top")
    # 3D
    a3=fig.add_axes([0.6,0.0,0.38,0.9],projection="3d")
    P=np.array([[25,120,8],[62,340,12],[31,95,4],[58,310,10]]); nm=["Ana","Bruno","Carla","Diego"]
    a3.scatter(P[:,0],P[:,1],P[:,2],c=[CYA,AMB,CYA,AMB],s=50)
    for p,n in zip(P,nm): a3.text(p[0],p[1],p[2]+0.8,n,fontsize=8)
    a3.set_xlabel("Idade",fontsize=7.5,labelpad=-6); a3.set_ylabel("Gasto",fontsize=7.5,labelpad=-6); a3.set_zlabel("Visitas",fontsize=7.5,labelpad=-6)
    a3.tick_params(labelsize=6,pad=-4); fig.text(0.79,0.93,"Cada linha é um ponto no espaço",ha="center",fontsize=9,fontweight="bold")
    save(fig,"f2_matriz",tight=False)

def f2_distancias():
    fig,ax=plt.subplots(figsize=(5.2,4.6)); ax.set_xlim(-0.5,8.5); ax.set_ylim(-0.5,6.5)
    for i in range(9): ax.axhline(i-0.5+0.5,c=MID,lw=0.7,zorder=0) if i<7 else None
    for i in range(9): ax.axvline(i,c=MID,lw=0.7,zorder=0)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    A=(1,1); B=(7,5)
    ax.plot([A[0],B[0]],[A[1],B[1]],c=IND,lw=2.8,label="Euclidiana: √(6²+4²) ≈ 7,2")
    ax.plot([1,7,7],[1,1,5],c=AMB,lw=2.8,ls="-",label="Manhattan: 6 + 4 = 10")
    ax.plot([1,5,7],[1,5,5],c=AMB,lw=1.2,ls=":")
    ax.plot([A[0],B[0]],[A[1],B[1]],c=GRN,lw=0,label="Chebyshev: máx(6, 4) = 6")
    ax.add_patch(Rectangle((1,1),6,4,fc="none",ec=GRN,lw=1.8,ls="--"))
    ax.scatter(*zip(A,B),s=110,c=[RED,RED],zorder=5,edgecolors="white"); ax.text(0.6,0.4,"A",fontsize=11,fontweight="bold"); ax.text(7.2,5.2,"B",fontsize=11,fontweight="bold")
    ax.legend(loc="upper left",fontsize=8,frameon=True,framealpha=0.95)
    for s in ax.spines.values(): s.set_visible(True)
    save(fig,"f2_distancias")

def f2_bolas():
    fig,ax=plt.subplots(1,3,figsize=(8.4,2.9))
    th=np.linspace(0,2*np.pi,400)
    for a,(t,c) in zip(ax,[("Manhattan (p=1)\nlosango",AMB),("Euclidiana (p=2)\ncírculo",IND),("Chebyshev (p→∞)\nquadrado",GRN)]):
        a.set_xlim(-1.4,1.4); a.set_ylim(-1.4,1.4); a.set_aspect("equal"); a.set_title(t,fontsize=9); a.axhline(0,c=MID,lw=1); a.axvline(0,c=MID,lw=1); clean(a,True)
    ax[0].fill([1,0,-1,0],[0,1,0,-1],fc=AMB,alpha=0.3,ec=AMB,lw=2.5)
    ax[1].add_patch(Circle((0,0),1,fc=IND,alpha=0.3,ec=IND,lw=2.5))
    ax[2].add_patch(Rectangle((-1,-1),2,2,fc=GRN,alpha=0.3,ec=GRN,lw=2.5))
    for a in ax: a.scatter([0],[0],c=INK,s=25,zorder=5); a.text(0.05,-0.18,"centro",fontsize=7)
    save(fig,"f2_bolas")

def f2_cosseno():
    fig,ax=plt.subplots(figsize=(5.4,4.2)); ax.set_xlim(-0.5,10); ax.set_ylim(-0.5,8)
    ax.annotate("",xy=(2,1.6),xytext=(0,0),arrowprops=dict(arrowstyle="-|>",color=IND,lw=3)); ax.text(2.1,1.0,"Texto curto (a)",color=IND,fontsize=8.5,fontweight="bold")
    ax.annotate("",xy=(8,6.4),xytext=(0,0),arrowprops=dict(arrowstyle="-|>",color=GRN,lw=3)); ax.text(5.2,7.1,"Texto longo (b)",color=GRN,fontsize=8.5,fontweight="bold")
    ax.annotate("",xy=(6.5,1),xytext=(0,0),arrowprops=dict(arrowstyle="-|>",color=RED,lw=3)); ax.text(6.6,0.2,"Outro assunto (c)",color=RED,fontsize=8.5,fontweight="bold")
    ax.plot([2,8],[1.6,6.4],c=GRY,ls="--"); ax.text(5.6,3.2,"distância euclidiana\nenorme",fontsize=8,color=GRY,rotation=0)
    th=np.linspace(np.arctan2(1,6.5),np.arctan2(6.4,8),50); ax.plot(2.2*np.cos(th),2.2*np.sin(th),c=INK,lw=1.2)
    ax.text(2.5,2.1,"ângulo\nmaior",fontsize=8)
    ax.text(0.2,5.3,"a e b: mesma direção\n→ cosseno = 1 (iguais)\na e c: ângulos diferentes\n→ cosseno menor",fontsize=8.5,bbox=dict(fc=LIGHT,ec=MID,boxstyle="round"))
    ax.set_xlabel("palavra X"); ax.set_ylabel("palavra Y"); ax.set_xticks([]); ax.set_yticks([])
    save(fig,"f2_cosseno")

def f2_escala():
    rng=np.random.default_rng(0)
    A=np.r_[rng.normal(2,0.8,100),rng.normal(8,0.8,100)]; B=rng.normal(500,260,200)
    X=np.c_[A,B]; y=np.r_[np.zeros(100),np.ones(100)]
    from sklearn.preprocessing import StandardScaler
    l1=KMeans(2,n_init=10,random_state=0).fit_predict(X); Xs=StandardScaler().fit_transform(X); l2=KMeans(2,n_init=10,random_state=0).fit_predict(Xs)
    fig,ax=plt.subplots(1,2,figsize=(8.4,3.6))
    pts(ax[0],X,[CL[i] for i in l1],20); ax[0].set_title("Sem escalar: o gasto domina\n(separação errada, horizontal)"); ax[0].set_xlabel("Visitas/mês (0 a 10)"); ax[0].set_ylabel("Gasto (R$, 0 a 1000)")
    ax[0].axhline(np.median(B),c=RED,ls="--",lw=1.2)
    pts(ax[1],Xs,[CL[i] for i in l2],20); ax[1].set_title("Padronizado: os grupos reais aparecem\n(separação correta, vertical)"); ax[1].set_xlabel("Visitas (padronizado)"); ax[1].set_ylabel("Gasto (padronizado)")
    ax[1].axvline(0,c=GRN,ls="--",lw=1.2)
    fig.tight_layout(); save(fig,"f2_escala")

def f2_correlacao():
    d=load_iris(); C=np.corrcoef(d.data.T); nm=["sépala\ncompr.","sépala\nlarg.","pétala\ncompr.","pétala\nlarg."]
    fig,ax=plt.subplots(figsize=(4.6,4.0)); im=ax.imshow(C,cmap="RdBu_r",vmin=-1,vmax=1)
    ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.set_xticklabels(nm,fontsize=8); ax.set_yticklabels(nm,fontsize=8)
    for i in range(4):
        for j in range(4): ax.text(j,i,f"{C[i,j]:.2f}",ha="center",va="center",color="white" if abs(C[i,j])>0.6 else INK,fontsize=9,fontweight="bold")
    for s in ax.spines.values(): s.set_visible(False)
    plt.colorbar(im,fraction=0.046,pad=0.04,label="correlação"); ax.set_title("Atributos correlacionados = informação repetida",fontsize=9)
    save(fig,"f2_correlacao")

def f2_maldicao():
    rng=np.random.default_rng(0); ds=[1,2,3,5,10,20,50,100,300,1000,3000]
    cont=[]
    for d in ds:
        X=rng.random((500,d)); dist=np.linalg.norm(X[1:]-X[0],axis=1); cont.append((dist.max()-dist.min())/dist.min())
    fig,ax=plt.subplots(1,2,figsize=(8.6,3.5),gridspec_kw={"width_ratios":[1.15,1]})
    ax[0].plot(ds,cont,"o-",c=IND,lw=2.2); ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("Número de dimensões"); ax[0].set_ylabel("(maior − menor distância) / menor")
    ax[0].set_title("O contraste entre distâncias desaba"); ax[0].grid(alpha=0.3)
    for d,c,col in [(2,None,CYA),(1000,None,RED)]:
        X=rng.random((1500,d)); dist=np.linalg.norm(X[1:]-X[0],axis=1); dist=dist/dist.mean()
        ax[1].hist(dist,bins=40,alpha=0.65,color=col,label=f"d = {d}")
    ax[1].set_title("Distâncias (relativas à média)"); ax[1].legend(); ax[1].set_xlabel("distância ÷ média"); ax[1].set_yticks([])
    fig.tight_layout(); save(fig,"f2_maldicao")

def f3_passos():
    X,_=make_blobs(150,centers=[[0,0],[4,1],[2,4.5]],cluster_std=0.7,random_state=5)
    C0=np.array([[-1.5,1.0],[2.5,-1.0],[1.0,3.0]])
    asg=lambda C: np.argmin(((X[:,None]-C[None])**2).sum(2),1)
    mean=lambda l: np.array([X[l==k].mean(0) for k in range(3)])
    l1=asg(C0); C1=mean(l1); C=C1
    for _ in range(10): C=mean(asg(C))
    lf=asg(C)
    fig,ax=plt.subplots(1,4,figsize=(11,3.0))
    tit=["1. Início:\ncentros aleatórios","2. Atribuir ao\ncentro mais próximo","3. Mover centros\npara a média","4. Repetir até\nestabilizar"]
    for it,a in enumerate(ax):
        if it==0: pts(a,X,GRY,16); a.scatter(*C0.T,marker="X",s=150,c=CL[:3],edgecolors=INK,zorder=5)
        elif it==1: pts(a,X,[CL[l] for l in l1],16); a.scatter(*C0.T,marker="X",s=150,c=CL[:3],edgecolors=INK,zorder=5)
        elif it==2:
            pts(a,X,[CL[l] for l in l1],16)
            for k in range(3): a.annotate("",xy=C1[k],xytext=C0[k],arrowprops=dict(arrowstyle="-|>",color=CL[k],lw=2))
            a.scatter(*C0.T,marker="o",s=70,facecolors="white",edgecolors=INK,zorder=4); a.scatter(*C1.T,marker="X",s=150,c=CL[:3],edgecolors=INK,zorder=5)
        else: pts(a,X,[CL[l] for l in lf],16); a.scatter(*C.T,marker="X",s=150,c=CL[:3],edgecolors=INK,zorder=5)
        a.set_title(tit[it],fontsize=9); clean(a,True)
    fig.tight_layout(); save(fig,"f3_passos")

def f3_voronoi():
    X,_=make_blobs(200,centers=4,cluster_std=1.1,random_state=4); km=KMeans(4,n_init=10,random_state=0).fit(X)
    xx,yy=np.meshgrid(np.linspace(X[:,0].min()-2,X[:,0].max()+2,400),np.linspace(X[:,1].min()-2,X[:,1].max()+2,400))
    Z=km.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
    from matplotlib.colors import ListedColormap
    fig,ax=plt.subplots(figsize=(5.2,4.2)); ax.contourf(xx,yy,Z,levels=[-.5,.5,1.5,2.5,3.5],colors=[IND,AMB,GRN,RED],alpha=0.16); ax.contour(xx,yy,Z,levels=[.5,1.5,2.5],colors=INK,linewidths=1.3)
    pts(ax,X,[CL[l] for l in km.labels_],20); ax.scatter(*km.cluster_centers_.T,marker="X",s=170,c=CL[:4],edgecolors=INK,zorder=5)
    ax.set_title("Células de Voronoi: cada centro domina uma região"); clean(ax,True)
    save(fig,"f3_voronoi")

def f3_init():
    X,_=make_blobs(300,centers=[[0,0],[5,0],[2.5,4.5]],cluster_std=0.6,random_state=2)
    bad=np.array([[2.082640083010488, 5.1681338870954745], [2.499270695813529, 1.9902934055091244], [3.3414941641076448, 3.925237494503894]])
    km1=KMeans(3,init=bad,n_init=1).fit(X)
    km2=KMeans(3,init="k-means++",n_init=1,random_state=1).fit(X)
    fig,ax=plt.subplots(1,2,figsize=(8.6,3.5))
    for a,km,t,c0 in [(ax[0],km1,f"Má inicialização\n(inércia = {km1.inertia_:.0f})",bad),(ax[1],km2,f"k-means++\n(inércia = {km2.inertia_:.0f})",None)]:
        pts(a,X,[CL[l] for l in km.labels_],16); a.scatter(*km.cluster_centers_.T,marker="X",s=160,c=CL[:3],edgecolors=INK,zorder=5)
        if c0 is not None: a.scatter(*c0.T,marker="o",s=90,facecolors="none",edgecolors=RED,lw=2,zorder=6)
        a.set_title(t,fontsize=9.5); clean(a,True)
    ax[0].text(0.03,0.03,"círculos vermelhos = centros iniciais",transform=ax[0].transAxes,fontsize=7.5,color=RED)
    fig.tight_layout(); save(fig,"f3_init")

def f3_cotovelo():
    X,_=make_blobs(400,centers=4,random_state=3); ks=range(1,10); ine=[KMeans(k,n_init=10,random_state=0).fit(X).inertia_ for k in ks]
    fig,ax=plt.subplots(figsize=(5.8,3.6)); ax.plot(list(ks),ine,"o-",c=IND,lw=2.4,ms=7)
    ax.scatter([4],[ine[3]],s=260,facecolors="none",edgecolors=RED,lw=2.5); ax.annotate("cotovelo\n(k = 4)",xy=(4,ine[3]),xytext=(5.6,ine[1]*0.8),arrowprops=dict(arrowstyle="->",color=RED),color=RED,fontweight="bold")
    ax.set_xlabel("Número de grupos (k)"); ax.set_ylabel("Inércia"); ax.grid(alpha=0.3); ax.set_title("Método do cotovelo")
    save(fig,"f3_cotovelo")

def f3_falhas():
    fig,ax=plt.subplots(1,3,figsize=(10,3.3))
    Xm,_=make_moons(300,noise=0.06,random_state=0); Xc,_=make_circles(300,factor=0.4,noise=0.05,random_state=0)
    Xa,_=make_blobs(300,centers=3,random_state=3); Xa=Xa@np.array([[0.6,-0.65],[-0.15,0.85]])
    for a,X,t in [(ax[0],Xm,"Luas (não convexas)"),(ax[1],Xc,"Anéis concêntricos"),(ax[2],Xa,"Elipses alongadas")]:
        l=KMeans(2 if a is not ax[2] else 3,n_init=10,random_state=0).fit_predict(X); pts(a,X,[CL[i] for i in l],14); a.set_title(t,fontsize=9.5); clean(a,True)
    fig.suptitle("K-means impõe 'bolas' — e erra quando os grupos têm outra forma",fontsize=10,y=1.02,fontweight="bold")
    fig.tight_layout(); save(fig,"f3_falhas")

def f3_compressao():
    h,w=120,180
    yy,xx=np.mgrid[0:h,0:w]
    img=np.zeros((h,w,3))
    sky=np.clip(yy/h,0,1)
    img[...,0]=0.35+0.55*sky; img[...,1]=0.55+0.25*sky; img[...,2]=0.95-0.35*sky
    sun=((xx-140)**2+(yy-28)**2)<18**2; img[sun]=[1,0.85,0.2]
    m1=(yy>h-(np.abs(xx-50)*0.9)-10-0)&(yy>55); img[(yy>=95-(55-np.abs(xx-50)*0.85).clip(0))&(yy<100)&(np.abs(xx-50)<65)]=[0.45,0.38,0.5]
    img[(yy>=100)]=[0.2,0.62,0.28]+0.12*np.sin(xx[yy>=100]/6.0)[:,None]*np.array([0.5,1,0.4])
    img=np.clip(img,0,1)
    px=img.reshape(-1,3)
    fig,ax=plt.subplots(1,4,figsize=(11,2.7))
    ax[0].imshow(img); ax[0].set_title("Original\n(milhares de cores)",fontsize=9)
    for a,k in zip(ax[1:],[32,8,3]):
        km=KMeans(k,n_init=2,random_state=0).fit(px); q=km.cluster_centers_[km.labels_].reshape(h,w,3); a.imshow(np.clip(q,0,1)); a.set_title(f"K-means com {k} cores",fontsize=9)
    for a in ax: clean(a)
    fig.tight_layout(); save(fig,"f3_compressao")

if __name__=="__main__":
    for n,f in list(globals().items()):
        if n.startswith("f") and n[1].isdigit() and callable(f): f()
