from fc import *
import json

def f11_casa():
    fig,ax=canvas(8.6,3.8)
    ax.text(50,97,"Arquitetura de rede = planta da casa",ha="center",fontsize=11,fontweight="bold")
    rbox(ax,2,30,40,56,"",fc="white",ec=IND,lw=2.2,r=2)
    ax.text(22,80,"A casa",ha="center",fontsize=9.5,fontweight="bold",color=IND)
    ax.add_patch(Polygon([[10,60],[22,72],[34,60]],closed=True,fc=AMB,ec=AMB))
    for i,(x,y) in enumerate([(10,38),(22,38),(10,49),(22,49)]): ax.add_patch(Rectangle((x,y),11,10,fc="#EEF2FF",ec=IND,lw=1.5))
    ax.text(22,33,"andares, cômodos, ligações",ha="center",fontsize=7.8,color=INK)
    rbox(ax,58,30,40,56,"",fc="white",ec=GRN,lw=2.2,r=2)
    ax.text(78,80,"A rede neural",ha="center",fontsize=9.5,fontweight="bold",color=GRN)
    xs=[64,74,84,94]; ns=[3,4,4,2]
    pts=[[(x,58+ (k-(n-1)/2)*7) for k in range(n)] for x,n in zip(xs,ns)]
    for a,b in zip(pts[:-1],pts[1:]):
        for p in a:
            for q in b: ax.plot([p[0],q[0]],[p[1],q[1]],color=GRY,lw=.6,zorder=1)
    for col in pts:
        for p in col: ax.add_patch(Circle(p,2.1,fc=GRN,ec="white",zorder=3))
    ax.text(78,33,"camadas, neurônios, ativação",ha="center",fontsize=7.8,color=INK)
    ax.text(50,56,"≈",ha="center",fontsize=22,color=GRY)
    ax.text(50,16,"Treinar = construir a casa. Uma planta ruim não é salva por pedreiros excelentes.",ha="center",fontsize=8.6,color=INK)
    ax.text(50,6,"NAS: deixar o computador procurar a planta.",ha="center",fontsize=8.6,color=GRY,style="italic")
    save(fig,"f11_casa")

def f11_pilares():
    fig,ax=canvas(8.6,3.9)
    ax.text(50,97,"Os três pilares do NAS (Elsken; Metzen; Hutter, 2019)",ha="center",fontsize=11,fontweight="bold")
    P=[("1. Espaço de busca","onde procurar:\nquais arquiteturas\nsão permitidas",IND),("2. Estratégia de busca","como procurar:\naleatória, evolução,\nreforço, gradiente",CYA),("3. Estimativa de\ndesempenho","como avaliar\ndepressa cada\ncandidata",GRN)]
    for i,(t,d,c) in enumerate(P):
        x=2+i*33
        rbox(ax,x,60,30,26,t,fc=c,ec=c,fs=9,tc="white",bold=True,r=2)
        rbox(ax,x,22,30,34,d,fc="white",ec=c,fs=8.6,tc=INK,lw=2,r=2)
    arrow(ax,(32.5,73),(35.5,73),lw=2,ms=11); arrow(ax,(65.5,73),(68.5,73),lw=2,ms=11)
    ax.text(50,10,"Mesma estrutura do AutoML clássico: espaço, estratégia e avaliação barata.",ha="center",fontsize=8.8,color=GRY,style="italic")
    save(fig,"f11_pilares")

def f11_espaco():
    fig,ax=canvas(8.6,4.2)
    ax.text(50,97,"Três jeitos de definir o espaço de arquiteturas",ha="center",fontsize=11,fontweight="bold")
    # cadeia
    ax.text(17,86,"Em cadeia",ha="center",fontsize=9.5,fontweight="bold",color=IND)
    for i in range(5): rbox(ax,5+i*0.0,70-i*11,24,8,["entrada","camada 1","camada 2","camada 3","saída"][i],fc=[GRY,IND,IND,IND,GRN][i],ec="white",fs=7.5,tc="white",bold=True,r=1)
    for i in range(4): arrow(ax,(17,70-i*11),(17,67.2-i*11+0.0),lw=1.4,ms=7)
    ax.text(17,10,"a busca escolhe nº e tipo\nde cada camada",ha="center",fontsize=7.8,color=INK)
    # celulas
    ax.text(50,86,"Baseado em células",ha="center",fontsize=9.5,fontweight="bold",color=CYA)
    rbox(ax,36,52,28,26,"",fc="#ECFEFF",ec=CYA,lw=2,r=2); ax.text(50,71,"célula (desenhada pela busca)",ha="center",fontsize=6.6,color=CYA,fontweight="bold")
    for p in [(42,57),(50,64),(58,57)]: ax.add_patch(Circle(p,2.2,fc=CYA,ec="white"))
    ax.plot([42,50,58],[57,64,57],color=CYA,lw=1.2); ax.plot([42,58],[57,57],color=CYA,lw=1.2)
    for i in range(3):
        rbox(ax,40,36-i*0,20,0.1,"",fc="white",ec="white") if False else None
    for i,x in enumerate([37,48,59]): rbox(ax,x,28,10,10,"célula",fc=CYA,ec="white",fs=6.5,tc="white",bold=True,r=1)
    arrow(ax,(47.2,33),(47.9,33),lw=1.2,ms=6); arrow(ax,(58.2,33),(58.9,33),lw=1.2,ms=6)
    ax.text(50,20,"a mesma célula é repetida",ha="center",fontsize=7.8,color=INK); ax.text(50,10,"e empilhada (Zoph et al., 2018)",ha="center",fontsize=7.8,color=INK)
    # hierarquico
    ax.text(83,86,"Hierárquico",ha="center",fontsize=9.5,fontweight="bold",color=GRN)
    rbox(ax,68,40,30,38,"",fc="#ECFDF5",ec=GRN,lw=2,r=2)
    for i in range(3): rbox(ax,71+i*9,60,8,13,"",fc="white",ec=GRN,lw=1.5,r=1)
    for i in range(3):
        for j in range(2): ax.add_patch(Rectangle((72+i*9+0.5+j*3.2,64-0*1),2.4,3,fc=GRN,ec="none"))
    rbox(ax,71,44,26,10,"bloco de blocos",fc=GRN,ec=GRN,fs=7.5,tc="white",bold=True,r=1)
    ax.text(83,22,"células feitas de\ncélulas menores,\nem vários níveis",ha="center",fontsize=7.8,color=INK)
    save(fig,"f11_espaco")

def f11_mapa():
    T=json.load(open("/home/user/claude/automl/build/f11_cache.json")); P=np.array([r["params"] for r in T]); A=np.array([r["v40"] for r in T])
    fig,ax=plt.subplots(figsize=(6.8,3.7))
    L=np.array([len(r["arq"][0]) for r in T])
    for l,c in zip((1,2,3),(IND,AMB,GRN)): ax.scatter(P[L==l],A[L==l],s=16,color=c,alpha=.55,label=f"{l} camada(s) oculta(s)",zorder=3)
    F=[i for i in range(len(T)) if not any((A[j]>=A[i] and P[j]<=P[i]) and (A[j]>A[i] or P[j]<P[i]) for j in range(len(T)))]; F.sort(key=lambda i:P[i])
    ax.plot(P[F],A[F],"-",color=VIO,lw=2.2,zorder=4,label="fronteira de Pareto"); ax.scatter(P[F],A[F],s=30,color=VIO,zorder=5)
    b=int(A.argmax()); ax.scatter([P[b]],[A[b]],s=130,facecolor="none",edgecolor=RED,lw=2,zorder=6); ax.annotate("melhor da validação",(P[b],A[b]),(P[b]*0.35,A[b]-0.12),fontsize=8,color=RED,arrowprops=dict(arrowstyle="->",color=RED))
    ax.set_xscale("log"); ax.set_ylim(0.5,1.0); ax.set_xlabel("número de parâmetros (escala logarítmica)"); ax.set_ylabel("acurácia de validação")
    ax.set_title("As 310 arquiteturas do mini-NAS (Digits)",fontsize=10); ax.legend(frameon=False,fontsize=7.5,loc="lower right"); ax.grid(color=MID,lw=.5)
    save(fig,"f11_mapa")

def f11_estrategias():
    fig,ax=canvas(8.6,4.3)
    ax.text(50,97,"Quatro estratégias de busca de arquitetura",ha="center",fontsize=11,fontweight="bold")
    S=[("Aleatória","sorteia arquiteturas\ne fica com a melhor\n(linha de base forte)",GRY),("Reforço","controlador gera\narquiteturas; a nota\nvira recompensa\n(Zoph; Le, 2017)",IND),("Evolução","sorteia, escolhe a\nmelhor, muta; a mais\nvelha sai\n(Real et al., 2019)",GRN),("Gradiente","escolha contínua entre\noperações, otimizada\npor gradiente\n(DARTS)",AMB)]
    for i,(t,d,c) in enumerate(S):
        x=1.5+(i%2)*49.5; y=52 if i<2 else 6
        rbox(ax,x,y,47,38,"",fc="white",ec=c,lw=2.2,r=2)
        ax.text(x+23.5,y+31,t,ha="center",fontsize=10,fontweight="bold",color=c)
        ax.text(x+23.5,y+14,d,ha="center",va="center",fontsize=8.2,color=INK,linespacing=1.35)
    save(fig,"f11_estrategias")

def f11_busca():
    d=json.load(open("/home/user/claude/automl/build/f11_busca.json")); fig,ax=plt.subplots(figsize=(6.6,3.5))
    for (n,c) in (("aleatória",GRY),("evolução",GRN)):
        C=np.array(d["curvas"][n]); x=np.arange(1,C.shape[1]+1)
        ax.plot(x,np.median(C,0),color=c,lw=2.4,label=n); ax.fill_between(x,np.percentile(C,25,0),np.percentile(C,75,0),color=c,alpha=.2)
    ax.set_xlabel("arquiteturas avaliadas"); ax.set_ylabel("melhor acurácia de validação"); ax.set_title("Aleatória contra evolução regularizada (300 repetições)",fontsize=10)
    ax.legend(frameon=False,fontsize=8,loc="lower right"); ax.grid(color=MID,lw=.5); ax.set_ylim(0.88,0.98)
    save(fig,"f11_busca")

def f11_estimativas():
    fig,ax=canvas(8.6,3.9)
    ax.text(50,97,"Estimar a qualidade sem treinar até o fim",ha="center",fontsize=11,fontweight="bold")
    E=[("Treino curto","poucas épocas, menos\ndados ou imagens\nmenores (Cap. 6)",IND),("Pesos\ncompartilhados","reaproveita pesos\nentre candidatas\n(Pham et al., 2018)",GRN),("Extrapolar a\ncurva","olha o começo da\ncurva de aprendizado\ne prevê o final",AMB),("Preditor de\ndesempenho","um modelo prevê a nota\npela descrição da\narquitetura (Cap. 9)",VIO)]
    for i,(t,d,c) in enumerate(E):
        x=1.5+i*24.7
        rbox(ax,x,58,22.5,28,t,fc=c,ec=c,fs=8.6,tc="white",bold=True,r=2)
        rbox(ax,x,16,22.5,38,d,fc="white",ec=c,fs=7.5,tc=INK,lw=2,r=2)
    ax.text(50,6,"Toda estimativa barata tem erro: o que importa é preservar a ordem das candidatas.",ha="center",fontsize=8.4,color=GRY,style="italic")
    save(fig,"f11_estimativas")

def f11_fidelidade():
    d=json.load(open("/home/user/claude/automl/build/f11_fid.json")); a=np.array(d["v10"]); b=np.array(d["v40"])
    fig,ax=plt.subplots(figsize=(5.4,4.0)); ax.scatter(a,b,s=16,color=IND,alpha=.55); ax.plot([0.1,1],[0.1,1],"--",color=GRY,lw=1)
    ax.set_xlabel("acurácia com 10 iterações (barata)"); ax.set_ylabel("acurácia com 40 iterações"); ax.set_title(f"Spearman = {d['rho']:.2f}: a ordem se preserva bem",fontsize=10)
    ax.grid(color=MID,lw=.5); save(fig,"f11_fidelidade")

if __name__=="__main__":
    for fn_ in (f11_casa,f11_pilares,f11_espaco,f11_mapa,f11_estrategias,f11_busca,f11_estimativas,f11_fidelidade):
        try: fn_()
        except FileNotFoundError as e: print("pendente",fn_.__name__,e)
