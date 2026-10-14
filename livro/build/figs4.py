from fc import *
from sklearn.datasets import make_blobs, load_digits
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import MinMaxScaler

def f10_cesta():
    fig,ax=canvas(9,3.8,(0,100),(0,42))
    items={"pão":AMB,"leite":CYA,"manteiga":"#FACC15","café":"#92400E"}
    bk=[["pão","leite"],["pão","manteiga","leite"],["pão","manteiga","café"],["leite","café"]]
    for i,b in enumerate(bk):
        x=2+i*24.5; rbox(ax,x,4,22,22,fc="#FEF3C7",ec=AMB,lw=2,r=4); ax.text(x+11,29,f"Compra {i+1}",ha="center",fontsize=9,fontweight="bold")
        for k,it in enumerate(b): rbox(ax,x+2,19-k*5.5,18,4.6,it,fc=items[it],tc="white" if it!="manteiga" else INK,bold=True,fs=8,r=1.5)
    ax.text(50,38,"Cada compra é um conjunto de itens. Quais itens aparecem juntos?",ha="center",fontsize=10,fontweight="bold")
    ax.text(50,0.3,"pão + leite aparecem juntos em 2 das 4 compras  →  suporte = 50 %",ha="center",fontsize=9,color=IND,fontweight="bold")
    save(fig,"f10_cesta")

def f10_metricas():
    fig,ax=canvas(9,4,(0,100),(0,46))
    ax.add_patch(Rectangle((2,3),44,40,fc="white",ec=INK,lw=1.6)); ax.text(3,40,"100 compras",fontsize=8,color=INK)
    ax.add_patch(Circle((18,22),13,fc=AMB,alpha=0.45,ec=AMB,lw=2)); ax.add_patch(Circle((30,22),11,fc=CYA,alpha=0.45,ec=CYA,lw=2))
    ax.text(12,22,"A\nsó A",ha="center",va="center",fontsize=8.5); ax.text(36,22,"B\nsó B",ha="center",va="center",fontsize=8.5); ax.text(24.5,22,"A e B",ha="center",va="center",fontsize=9,fontweight="bold")
    ax.text(18,36.5,"A: fraldas (40)",ha="center",fontsize=8.5,color="#B45309",fontweight="bold"); ax.text(32,34,"B: cerveja (30)",ha="center",fontsize=8.5,color="#0E7490",fontweight="bold")
    ax.text(24.5,6,"interseção A∩B: 20 compras",ha="center",fontsize=8.5,color=INK)
    rows=[("Suporte(A→B)","20 / 100 = 0,20",IND),("Confiança","20 / 40 = 0,50   P(B | A)",GRN),("Lift","0,50 / 0,30 = 1,67",RED)]
    for i,(t,v,c) in enumerate(rows):
        rbox(ax,52,30-i*13,46,10.5,"",fc="white",ec=c,lw=2,r=2.5); ax.text(54,36-i*13,t,fontsize=10,fontweight="bold",color=c,va="center"); ax.text(54,32.2-i*13,v,fontsize=9.5,va="center")
    ax.text(75,2,"lift > 1: aparecem juntos mais que o acaso",ha="center",fontsize=8.5,style="italic")
    save(fig,"f10_metricas")

def f10_reticulado():
    from itertools import combinations
    items="ABCD"; fig,ax=canvas(8.6,4.8,(0,100),(0,60)); infreq={"C"}
    lv=[[""]]+[["".join(c) for c in combinations(items,k)] for k in range(1,5)]; ys=[54,42,28,14,3]
    pos={}
    for L,row in enumerate(lv):
        n=len(row)
        for i,s in enumerate(row):
            x=(i+1)*100/(n+1); pos[s]=(x,ys[L])
    for L in range(1,5):
        for s in lv[L]:
            for t in lv[L-1]:
                if t!="" or True:
                    if set(t)<=set(s) and len(t)==len(s)-1: ax.plot([pos[s][0],pos[t][0]],[pos[s][1],pos[t][1]],c=MID,lw=1.2,zorder=0)
    for L,row in enumerate(lv):
        for s in row:
            pr=any(c in s for c in infreq); x,y=pos[s]; lab=s if s else "∅"
            rbox(ax,x-4.2,y-2.3,8.4,4.6,lab,fc="#FEE2E2" if pr else "#D1FAE5",ec=RED if pr else GRN,fs=8.5,bold=True,r=1.2,lw=1.5)
    ax.text(2,57.5,"vermelho = contém C (pouco frequente)\n→ descartado sem contar",fontsize=8.5,color=RED,fontweight="bold",va="top")
    ax.text(98,0.2,"verde = candidatos a avaliar",fontsize=8.5,color=GRN,fontweight="bold",ha="right")
    ax.text(98,6,"AB, AD, BD, ABD\nainda precisam\nser contados",fontsize=7.5,color=GRN,ha="right",va="bottom")
    save(fig,"f10_reticulado")

def f10_fptree():
    T=[["pão","leite"],["pão","manteiga","leite"],["pão","manteiga"],["leite","café"],["pão","leite","café"],["pão","manteiga","café"],["leite","manteiga"],["pão","manteiga","leite","café"],["pão","café"],["pão","leite","manteiga"]]
    order={"pão":0,"leite":1,"manteiga":2,"café":3}; root={"n":"raiz","c":0,"ch":{}}
    for t in T:
        node=root
        for it in sorted(t,key=lambda x:order[x]):
            node=node["ch"].setdefault(it,{"n":it,"c":0,"ch":{}}) if it in node["ch"] else node["ch"].setdefault(it,{"n":it,"c":0,"ch":{}}); node["c"]+=1
    def leaves(nd): return 1 if not nd["ch"] else sum(leaves(c) for c in nd["ch"].values())
    pos=[]; cnt=[0]
    def lay(nd,d):
        if not nd["ch"]: x=cnt[0]; cnt[0]+=1
        else:
            xs=[lay(c,d+1) for c in nd["ch"].values()]; x=np.mean(xs)
        nd["x"]=x; nd["d"]=d; return x
    lay(root,0); W=cnt[0]
    fig,ax=plt.subplots(figsize=(8.6,4.6)); ax.axis("off"); ax.set_xlim(-0.8,W-0.2); ax.set_ylim(-5.3,0.9)
    cm={"pão":AMB,"leite":CYA,"manteiga":"#FACC15","café":"#92400E","raiz":INK}
    def draw(nd):
        for c in nd["ch"].values():
            ax.plot([nd["x"],c["x"]],[-nd["d"],-c["d"]],c=GRY,lw=1.3,zorder=0); draw(c)
        ax.add_patch(FancyBboxPatch((nd["x"]-0.43,-nd["d"]-0.3),0.86,0.6,boxstyle="round,pad=0,rounding_size=0.15",fc=cm[nd["n"]],ec="white",lw=1.2))
        ax.text(nd["x"],-nd["d"],nd["n"]+("" if nd["n"]=="raiz" else f":{nd['c']}"),ha="center",va="center",fontsize=7.8,fontweight="bold",color="white" if nd["n"] not in ("manteiga",) else INK)
    draw(root); ax.text(-0.7,-5.1,"Itens ordenados por frequência: pão > leite > manteiga > café. Nó 'item:n' = n compras passam por aquele caminho.",fontsize=8,style="italic",color=GRY)
    save(fig,"f10_fptree")

def f10_rede():
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import fpgrowth, association_rules
    import pandas as pd
    pedidos=[["notebook","mouse","mochila"],["notebook","mouse"],["celular","capinha","película"],["celular","capinha"],["notebook","mochila"],["celular","película"],["notebook","mouse","teclado"],["celular","capinha","película","carregador"],["mouse","teclado"],["notebook","mouse","mochila","teclado"]]*5
    te=TransactionEncoder(); df=pd.DataFrame(te.fit(pedidos).transform(pedidos),columns=te.columns_)
    r=association_rules(fpgrowth(df,min_support=0.15,use_colnames=True),metric="lift",min_threshold=1.3)
    r=r[(r.antecedents.apply(len)==1)&(r.consequents.apply(len)==1)]
    nodes=sorted(set(te.columns_)); ang=np.linspace(0,2*np.pi,len(nodes),endpoint=False); P={n:(np.cos(a),np.sin(a)) for n,a in zip(nodes,ang)}
    fig,ax=plt.subplots(figsize=(6.2,5)); ax.set_xlim(-1.6,1.6); ax.set_ylim(-1.45,1.45); ax.set_aspect("equal"); ax.axis("off")
    for _,row in r.iterrows():
        a=list(row.antecedents)[0]; b=list(row.consequents)[0]
        ax.add_patch(FancyArrowPatch(P[a],P[b],arrowstyle="-|>",mutation_scale=14,lw=0.8+row["lift"]*1.1,color=VIO,alpha=0.55,connectionstyle="arc3,rad=0.15",shrinkA=16,shrinkB=16))
    for n,(x,y) in P.items():
        ax.add_patch(Circle((x,y),0.14,fc=IND,ec="white",lw=2)); ax.text(x*1.3 if abs(x)>0.1 else x,y*1.22,n,ha="center",va="center",fontsize=8.5,fontweight="bold")
    ax.set_title("Regras de associação como rede (espessura = lift)",fontsize=10); save(fig,"f10_rede")

def f11_agulha():
    rng=np.random.default_rng(0); X=rng.normal(0,1,(800,2)); A=np.array([[3.6,2.8],[-3.5,-2.6],[3.0,-3.1]])
    fig,ax=plt.subplots(figsize=(6,3.8)); pts(ax,X,GRY,12,alpha=0.7); ax.scatter(*A.T,c=RED,s=130,marker="X",edgecolors="white",zorder=5)
    for a in A: ax.add_patch(Circle(a,0.55,fc="none",ec=RED,lw=2))
    ax.text(0,-3.9,"3 anomalias entre 803 pontos",ha="center",color=RED,fontweight="bold"); clean(ax,True); ax.set_title("A agulha no palheiro")
    save(fig,"f11_agulha")

def f11_tipos():
    rng=np.random.default_rng(1); t=np.arange(240); y=10+4*np.sin(2*np.pi*t/60)+rng.normal(0,0.4,240)
    y[40]+=9; y[150:162]=y[150:162].mean()*0+np.linspace(6,6.2,12)+rng.normal(0,0.05,12); y[100]=y[100]-0  # coletiva: trecho plano
    y[200]=2.6  # contextual: valor normal em geral (2.6 ok em vale?) 
    fig,ax=plt.subplots(figsize=(8.6,3.4)); ax.plot(t,y,c=IND,lw=1.4)
    ax.scatter([40],[y[40]],s=130,c=RED,zorder=5); ax.annotate("pontual",xy=(40,y[40]),xytext=(48,y[40]),color=RED,fontweight="bold")
    ax.axvspan(150,162,color=AMB,alpha=0.3); ax.text(156,3.5,"coletiva\n(trecho 'travado')",color="#B45309",fontweight="bold",ha="center",va="center",fontsize=8.5)
    ax.scatter([205],[y[205]],s=0); i=int(np.argmax(y[190:215]))+190
    ax.scatter([88],[y[88]],s=0)
    ax.set_xlabel("Tempo"); ax.set_ylabel("Medida"); ax.set_ylim(1,19); ax.set_title("Três tipos de anomalia em uma série temporal",fontsize=10)
    ax.annotate("contextual: valor alto\nnum 'vale' sazonal",xy=(185,y[185]),xytext=(168,17.2),arrowprops=dict(arrowstyle="->",color=VIO),color=VIO,fontweight="bold",fontsize=8.5)
    save(fig,"f11_tipos")

def f11_iforest():
    rng=np.random.default_rng(3); N=rng.normal(0,1,(60,2)); a=np.array([[3.4,2.9]]); X=np.r_[N,a]
    fig,ax=plt.subplots(1,3,figsize=(10.4,3.5))
    def splits(point,seed,n):
        r=np.random.default_rng(seed); lo=np.array([-4,-3.5]); hi=np.array([4.6,3.8]); segs=[]
        for _ in range(n):
            ft=r.integers(2); v=r.uniform(lo[ft],hi[ft]); segs.append((ft,v,lo.copy(),hi.copy()))
            if point[ft]<v: hi[ft]=v
            else: lo[ft]=v
        return segs
    for a_,(pt,title,col,n) in zip(ax[:2],[(a[0],"Anomalia: isolada em 3 cortes",RED,3),(N[0],"Ponto normal: muitos cortes (≈ 9+)",IND,9)]):
        pts(a_,N,GRY,14)
        if col==RED: a_.scatter(*a.T,c=RED,s=110,marker="X",zorder=5)
        else: a_.scatter(*N[0],c=IND,s=80,zorder=5,edgecolors="white"); a_.scatter(*a.T,c=GRY,s=40,marker="x",zorder=4)
        for k,(ft,v,lo,hi) in enumerate(splits(pt,5 if col==RED else 2,n)):
            if ft==0: a_.plot([v,v],[lo[1],hi[1]],c=col,lw=1.3,alpha=0.8)
            else: a_.plot([lo[0],hi[0]],[v,v],c=col,lw=1.3,alpha=0.8)
        a_.set_xlim(-4,4.6); a_.set_ylim(-3.5,3.8); a_.set_title(title,fontsize=9.5); clean(a_,True)
    f=IsolationForest(n_estimators=200,random_state=0).fit(X); sc=-f.score_samples(X)
    ax[2].hist(sc[:-1],bins=18,color=IND,alpha=0.8,label="normais"); ax[2].axvline(sc[-1],c=RED,lw=3,label="anomalia"); ax[2].legend(frameon=False,fontsize=8); ax[2].set_title("Escore de anomalia",fontsize=9.5); ax[2].set_yticks([]); ax[2].set_xlabel("escore (maior = mais anômalo)")
    fig.tight_layout(); save(fig,"f11_iforest")

def f11_lof():
    rng=np.random.default_rng(0); D=rng.normal([0,0],0.3,(120,2)); S=rng.normal([4.5,3.5],1.2,(80,2)); o=np.array([[1.3,1.1]]); X=np.r_[D,S,o]
    l=LocalOutlierFactor(15).fit(X); sc=-l.negative_outlier_factor_
    fig,ax=plt.subplots(figsize=(6,4.2)); pts(ax,X[:-1],GRY,14); ax.scatter(*o.T,c=RED,s=140,marker="X",edgecolors="white",zorder=5)
    ax.add_patch(Circle(o[0],1.0,fc="none",ec=RED,ls="--",lw=1.5)); ax.add_patch(Circle([0,0],0.35,fc="none",ec=IND,ls="--",lw=1.5)); ax.add_patch(Circle(S[0],1.0,fc="none",ec=GRN,ls="--",lw=1.5))
    ax.text(1.0,2.6,f"LOF = {sc[-1]:.1f}\n(vizinhos bem mais densos)",color=RED,fontweight="bold",fontsize=8,ha="center"); ax.text(-0.4,-1.3,"LOF ≈ 1 (denso, normal)",color=IND,fontweight="bold",fontsize=8,ha="center"); ax.text(5.6,1.2,"LOF ≈ 1\n(esparso mas normal)",color=GRN,fontweight="bold",fontsize=8,ha="center")
    ax.set_aspect("equal"); ax.set_ylim(-2,6.2); clean(ax,True); ax.set_title("Densidade local importa"); save(fig,"f11_lof")

def _bound(ax,model,X,col):
    xx,yy=np.meshgrid(np.linspace(-6,6,250),np.linspace(-6,6,250)); Z=model.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx,yy,Z,levels=[-2,0,2],colors=["#FEE2E2","#E0E7FF"],alpha=0.8); ax.contour(xx,yy,Z,levels=[0],colors=col,linewidths=2)

def f11_ocsvm():
    rng=np.random.default_rng(0); X=rng.normal(0,1,(300,2)); T=rng.uniform(-5.5,5.5,(25,2))
    m=OneClassSVM(gamma=0.3,nu=0.05).fit(X); fig,ax=plt.subplots(figsize=(5.4,4.4)); _bound(ax,m,X,IND)
    pts(ax,X,IND,14); p=m.predict(T); ax.scatter(*T[p==-1].T,c=RED,s=80,marker="X",edgecolors="white",zorder=5); ax.scatter(*T[p==1].T,c=GRN,s=40,edgecolors="white",zorder=5)
    ax.set_xlim(-6,6); ax.set_ylim(-6,6); ax.set_aspect("equal"); clean(ax,True); ax.set_title("One-Class SVM: a 'cerca' em volta do normal"); save(fig,"f11_ocsvm")

def f11_comparacao():
    rng=np.random.default_rng(1); X=np.r_[rng.normal([-1.5,0],[0.9,0.6],(250,2)),rng.normal([2,1.5],0.5,(150,2))]; T=rng.uniform(-6,6,(30,2)); XX=np.r_[X,T]
    ms=[("Isolation Forest",IsolationForest(contamination=0.06,random_state=0).fit(X)),("LOF (novelty)",LocalOutlierFactor(25,novelty=True,contamination=0.06).fit(X)),("One-Class SVM",OneClassSVM(gamma=0.15,nu=0.06).fit(X)),("Envelope elíptico",EllipticEnvelope(contamination=0.06,random_state=0).fit(X))]
    fig,ax=plt.subplots(1,4,figsize=(12,3.2))
    for a,(n,m) in zip(ax,ms):
        xx,yy=np.meshgrid(np.linspace(-6,6,200),np.linspace(-6,6,200)); Z=m.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
        a.contourf(xx,yy,Z,levels=[-2,0,2],colors=["#FEE2E2","#E0E7FF"],alpha=0.85); a.contour(xx,yy,Z,levels=[0],colors=INK,linewidths=1.2)
        pts(a,X,IND,6); p=m.predict(T); a.scatter(*T[p==-1].T,c=RED,s=22,marker="X",zorder=5); a.scatter(*T[p==1].T,c=GRN,s=22,zorder=5)
        a.set_xlim(-6,6); a.set_ylim(-6,6); a.set_title(n,fontsize=9.5); clean(a,True)
    fig.tight_layout(); save(fig,"f11_comparacao")

def f11_limiar():
    rng=np.random.default_rng(0); n=rng.gamma(4,0.6,2000); a=rng.normal(7,1.2,40)
    fig,ax=plt.subplots(figsize=(6.6,3.5)); b=np.linspace(0,11,60); ax.hist(n,bins=b,color=IND,alpha=0.75,label="normais"); ax.hist(a,bins=b,color=RED,alpha=0.85,label="anomalias")
    th=5.2; ax.axvline(th,c=INK,ls="--",lw=2); ax.text(th+0.1,ax.get_ylim()[1]*0.85,"limiar",fontweight="bold"); ax.text(th+0.3,ax.get_ylim()[1]*0.5,"à direita: alertas\n(normais aqui = falsos alarmes)",ha="left",fontsize=8,color=IND); ax.text(th-0.3,ax.get_ylim()[1]*0.6,"à esquerda: anomalias\nperdidas (se houver)",ha="right",fontsize=8,color=RED)
    ax.set_xlabel("Escore de anomalia"); ax.set_yticks([]); ax.legend(frameon=False); ax.set_title("Mover o limiar troca falsos alarmes por anomalias perdidas"); save(fig,"f11_limiar")

def _net(ax,layers,x0,x1,col,ys=(8,92)):
    xs=np.linspace(x0,x1,len(layers)); allp=[]
    for x,n in zip(xs,layers):
        yy=np.linspace(*ys,n) if n>1 else [50]; allp.append([(x,y) for y in yy])
    for L in range(len(layers)-1):
        for p in allp[L]:
            for q in allp[L+1]: ax.plot([p[0],q[0]],[p[1],q[1]],c=MID,lw=0.5,zorder=0)
    for L,ps in enumerate(allp):
        ax.scatter([p[0] for p in ps],[p[1] for p in ps],s=170,c=col[L],edgecolors="white",linewidths=1.2,zorder=3)
    return xs

def f12_mapa():
    fig,ax=canvas(9,4.6,(0,100),(0,56))
    rbox(ax,32,46,36,8,"Aprendizado profundo\nnão supervisionado",fc=IND,tc="white",bold=True,fs=9.5,r=2.5)
    groups=[("Autoencoders",GRN,4,["Subcompleto","Esparso","Denoising","Convolucional"]),("Modelos generativos",AMB,36,["VAE","GAN","Difusão"]),("Auto-supervisionado",RED,68,["Tarefas-pretexto","Contrastivo\n(SimCLR, MoCo)","Sem negativos\n(BYOL, DINO)"])]
    for t,c,x,ch in groups:
        rbox(ax,x,30,28,8,t,fc=c,tc="white",bold=True,fs=9,r=2.5); arrow(ax,(50,46),(x+14,38.4),INK,lw=1.2)
        for i,s in enumerate(ch):
            rbox(ax,x+2,22-i*7.2,24,6,s,fc="white",ec=c,lw=1.6,fs=7.8,r=1.5); 
    save(fig,"f12_mapa")

def f12_ae_arquitetura():
    fig,ax=canvas(9,4,(0,100),(0,100)); cols=[IND]*1+[CYA]*1+[RED]+[CYA]+[IND]
    xs=_net(ax,[8,5,2,5,8],12,88,[IND,CYA,RED,CYA,GRN],(14,80))
    for x,t in zip(xs,["Entrada (x)","","Código\\n(espaço latente z)","","Reconstrução (x̂)"]): ax.text(x,88,t.replace("\\n","\n"),ha="center",fontsize=8.5,fontweight="bold",va="bottom" if False else "center")
    ax.annotate("",xy=(xs[2]-4,8),xytext=(xs[0]-2,8),arrowprops=dict(arrowstyle="-",color=CYA,lw=0)); 
    rbox(ax,xs[0]-3,0.5,xs[1]-xs[0]+8,7,"Codificador (comprime)",fc="#CFFAFE",ec=CYA,fs=8.5,r=2); rbox(ax,xs[3]-5,0.5,xs[4]-xs[3]+8,7,"Decodificador (reconstrói)",fc="#D1FAE5",ec=GRN,fs=8.5,r=2)
    ax.text(50,94.5,"perda = diferença entre x e x̂",ha="center",fontsize=8.5,color=RED,style="italic")
    save(fig,"f12_ae_arquitetura")

def _train_ae(dim=2,epochs=250,noise=0.0,seed=0):
    import torch; from torch import nn
    torch.manual_seed(seed); X=MinMaxScaler().fit_transform(load_digits().data).astype("float32"); Xt=torch.from_numpy(X)
    enc=nn.Sequential(nn.Linear(64,48),nn.ReLU(),nn.Linear(48,dim)); dec=nn.Sequential(nn.Linear(dim,48),nn.ReLU(),nn.Linear(48,64),nn.Sigmoid())
    opt=torch.optim.Adam(list(enc.parameters())+list(dec.parameters()),lr=5e-3)
    for _ in range(epochs):
        inp=(Xt+noise*torch.randn_like(Xt)).clamp(0,1) if noise else Xt
        l=nn.functional.mse_loss(dec(enc(inp)),Xt); opt.zero_grad(); l.backward(); opt.step()
    return enc,dec,Xt,load_digits().target

def f12_latente():
    import torch
    enc,dec,Xt,y=_train_ae(2,400)
    with torch.no_grad(): Z=enc(Xt).numpy()
    fig,ax=plt.subplots(figsize=(5.6,4.4)); s=ax.scatter(Z[:,0],Z[:,1],c=y,cmap="tab10",s=8); plt.colorbar(s,label="dígito (não usado no treino)"); clean(ax,True); ax.set_title("Espaço latente 2D de um autoencoder"); save(fig,"f12_latente")

def f12_denoising():
    import torch
    enc,dec,Xt,y=_train_ae(16,500,noise=0.35); torch.manual_seed(5)
    with torch.no_grad():
        noisy=(Xt+0.35*torch.randn_like(Xt)).clamp(0,1); rec=dec(enc(noisy))
    idx=[0,1,2,3,4,5,6,7]; fig,ax=plt.subplots(3,8,figsize=(8,3.4))
    for r,(M,t) in enumerate([(Xt,"limpa"),(noisy,"ruidosa"),(rec,"recuperada")]):
        for j,i in enumerate(idx): ax[r,j].imshow(M[i].reshape(8,8),cmap="gray_r"); clean(ax[r,j])
        ax[r,0].set_ylabel(t,rotation=0,ha="right",va="center",fontsize=9)
    fig.tight_layout(); save(fig,"f12_denoising")

def f12_vae():
    fig,ax=canvas(9.2,3.6,(0,100),(0,40))
    rbox(ax,1,15,10,10,"x",fc=IND,tc="white",bold=True,fs=12,r=2); rbox(ax,15,12,14,16,"Codificador",fc="#E0E7FF",ec=IND,fs=8.5,r=2)
    rbox(ax,34,23,10,7,"μ (média)",fc="#CFFAFE",ec=CYA,fs=8,r=1.5); rbox(ax,34,10,10,7,"σ (desvio)",fc="#CFFAFE",ec=CYA,fs=8,r=1.5)
    rbox(ax,50,4,10,6,"ε ~ N(0,1)",fc="#FEF3C7",ec=AMB,fs=8,r=1.5); rbox(ax,50,15,14,10,"z = μ + σ·ε",fc="#EDE9FE",ec=VIO,fs=9,bold=True,r=2)
    rbox(ax,69,12,14,16,"Decodificador",fc="#D1FAE5",ec=GRN,fs=8.5,r=2); rbox(ax,87,15,10,10,"x̂",fc=GRN,tc="white",bold=True,fs=12,r=2)
    for a,b in [((11,20),(15,20)),((29,22),(34,26)),((29,18),(34,14)),((44,26),(50,22)),((44,14),(50,18)),((55,10),(57,15)),((64,20),(69,20)),((83,20),(87,20))]: arrow(ax,a,b,INK,lw=1.3)
    ax.text(50,35,"A perda soma: erro de reconstrução + KL (aproxima N(μ,σ²) de N(0,1))",ha="center",fontsize=9,color=RED,fontweight="bold")
    g=np.linspace(-1,1,50); ax.plot(40+g*6,36-0+0*g,alpha=0)
    save(fig,"f12_vae")

def f12_gan():
    fig,ax=canvas(9,3.8,(0,100),(0,42))
    rbox(ax,1,15,13,10,"Ruído\nz aleatório",fc="#FEF3C7",ec=AMB,fs=8.5,r=2); rbox(ax,20,12,17,16,"GERADOR\n(falsificador)",fc=RED,tc="white",bold=True,fs=9,r=3)
    rbox(ax,43,27,14,8,"Imagem\nfalsa",fc="#FEE2E2",ec=RED,fs=8.5,r=2); rbox(ax,43,6,14,8,"Imagem\nreal",fc="#D1FAE5",ec=GRN,fs=8.5,r=2)
    rbox(ax,63,12,17,16,"DISCRIMINADOR\n(polícia)",fc=IND,tc="white",bold=True,fs=9,r=3); rbox(ax,86,17,13,7,"Real ou\nfalsa?",fc="white",ec=INK,fs=8.5,r=2)
    for a,b in [((14,20),(20,20)),((37,22),(43,30)),((57,31),(63,24)),((57,10),(63,16)),((80,20),(86,20.5))]: arrow(ax,a,b,INK,lw=1.4)
    arrow(ax,(71,11.5),(30,10.8),AMB,lw=1.6,rad=-0.45,ls="--"); ax.text(50,1.2,"feedback: o gerador aprende a enganar; o discriminador, a desmascarar",ha="center",fontsize=8.5,color="#B45309",fontweight="bold")
    save(fig,"f12_gan")

def f12_pretexto():
    fig,ax=canvas(10,3.4,(0,100),(0,36))
    rbox(ax,1,2,31,32,fc="white",ec=VIO,lw=2,r=3); rbox(ax,1,27,31,7,"1. Esconder palavras",fc=VIO,tc="white",bold=True,fs=9,r=3)
    ax.text(16.5,19,"“O gato subiu no ____”",ha="center",fontsize=9.5); ax.text(16.5,10,"alvo: telhado",ha="center",fontsize=9,color=VIO,fontweight="bold")
    rbox(ax,34.5,2,31,32,fc="white",ec=CYA,lw=2,r=3); rbox(ax,34.5,27,31,7,"2. Prever a rotação",fc=CYA,tc="white",bold=True,fs=9,r=3)
    for i,(rot,lab) in enumerate([(0,"0°"),(90,"90°"),(180,"180°"),(270,"270°")]):
        cx=39+i*7.4; 
        pg=np.array([[0,2],[1.4,-1.4],[-1.4,-1.4]]); th=np.radians(rot); R=np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]]); q=pg@R.T*1.2
        ax.add_patch(Polygon(q+[cx+2.5,17],fc=AMB,ec=INK)); ax.text(cx+2.5,8.5,lab,ha="center",fontsize=8)
    ax.text(50,3.5,"alvo: o ângulo aplicado",ha="center",fontsize=8.5,color=CYA,fontweight="bold")
    rbox(ax,68,2,31,32,fc="white",ec=GRN,lw=2,r=3); rbox(ax,68,27,31,7,"3. Mascarar partes",fc=GRN,tc="white",bold=True,fs=9,r=3)
    for i in range(4):
        for j in range(4): ax.add_patch(Rectangle((75+i*4.4,7+j*4.4),4.2,4.2,fc=MID if (i*3+j*5)%4==0 else "#A7F3D0",ec="white"))
    save(fig,"f12_pretexto")

def f12_contrastivo():
    rng=np.random.default_rng(2); fig,ax=plt.subplots(figsize=(6,4.4))
    cen={"gato":(0,0),"carro":(4,2.2),"flor":(0.5,4)}; cols={"gato":IND,"carro":AMB,"flor":GRN}
    for n,c in cen.items():
        P=rng.normal(c,0.35,(10,2)); ax.scatter(*P.T,c=cols[n],s=30,edgecolors="white",alpha=0.8); ax.text(c[0],c[1]-0.85,n,ha="center",fontweight="bold",color=cols[n])
    a=(-0.2,0.15); b=(0.45,-0.1); ax.scatter(*zip(a,b),c=IND,s=110,edgecolors=INK,zorder=5); ax.annotate("",xy=b,xytext=(a[0]-0.05,a[1]+0.0),arrowprops=dict(arrowstyle="<->",color=GRN,lw=2.4)); ax.text(0.1,0.65,"positivos:\nse atraem",color=GRN,fontweight="bold",ha="center",fontsize=8.5)
    ax.annotate("",xy=(3.6,2.0),xytext=(0.6,0.2),arrowprops=dict(arrowstyle="<->",color=RED,lw=2.4,ls="--")); ax.text(2.2,0.7,"negativos: se repelem",color=RED,fontweight="bold",fontsize=8.5,rotation=30)
    ax.annotate("",xy=(0.5,3.5),xytext=(0.1,0.6),arrowprops=dict(arrowstyle="<->",color=RED,lw=2.4,ls="--"))
    ax.set_xlim(-1.5,5.5); ax.set_ylim(-1.5,5); clean(ax,True); ax.set_title("Espaço de representação (aprendizado contrastivo)"); save(fig,"f12_contrastivo")

def f12_fechamento():
    fig,ax=canvas(9.4,4.8,(0,100),(0,56))
    rbox(ax,30,23,40,11,"Aprendizado\nnão supervisionado",fc=IND,tc="white",bold=True,fs=11,r=4)
    cfg=[("AGRUPAR",CYA,1,36,"K-means • Hierárquico\nDBSCAN/HDBSCAN • GMM\nEspectral • Deep clustering"),("REDUZIR",GRN,1,2,"PCA • ICA • NMF\nIsomap • t-SNE • UMAP\nAutoencoders • VAE"),("ASSOCIAR",AMB,69,36,"Apriori • FP-Growth\nEclat • padrões\nsequenciais"),("DETECTAR O RARO",RED,69,2,"Isolation Forest • LOF\nOne-Class SVM • GMM\nAutoencoder • ECOD")]
    for t,c,x,y,d in cfg:
        rbox(ax,x,y,30,18,fc="white",ec=c,lw=2.2,r=3); rbox(ax,x,y+12,30,6,t,fc=c,tc="white",bold=True,fs=9.5,r=3); ax.text(x+15,y+5.8,d,ha="center",va="center",fontsize=7.8)
    for (x,y,tx,ty) in [(31,34,22,40),(31,23,22,17),(69,34,78,40),(69,23,78,17)]: arrow(ax,(x,y),(tx+8 if tx<50 else tx-8,ty),INK,lw=1.4)
    save(fig,"f12_fechamento")

if __name__=="__main__":
    for n,f in list(globals().items()):
        if n.startswith("f") and n[1].isdigit() and callable(f):
            try: f()
            except Exception as e: print("ERR",n,repr(e)[:300])
