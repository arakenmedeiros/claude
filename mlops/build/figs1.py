from fc import *
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Wedge, Arc

PART={1:(IND,CYA),2:(IND,CYA),3:(CYA,GRN),4:(CYA,GRN),5:(CYA,GRN),6:(CYA,GRN),7:(GRN,AMB),8:(GRN,AMB),9:(GRN,AMB),10:(AMB,PNK),11:(VIO,PNK)}

def lemniscate(t,a=1.0):
    x=a*np.cos(t)/(1+np.sin(t)**2); y=a*np.sin(t)*np.cos(t)/(1+np.sin(t)**2); return x,y

def banner(n):
    c1,c2=PART[n]; rng=np.random.default_rng(100+n)
    fig,ax=plt.subplots(figsize=(10,1.9)); fig.patch.set_facecolor(NAVY); ax.set_facecolor(NAVY); ax.set_xlim(0,10); ax.set_ylim(0,1.9); ax.axis("off")
    g=np.linspace(0,1,256)[None,:]; cm=LinearSegmentedColormap.from_list("b",[NAVY,"#1E1B4B",NAVY]); ax.imshow(g,extent=[0,10,0,1.9],aspect="auto",cmap=cm,zorder=0)
    # grade de pontos
    for gx in np.arange(0.3,10,0.35):
        for gy in np.arange(0.2,1.9,0.35):
            ax.plot(gx,gy,".",ms=1.2,c="white",alpha=0.08,zorder=1)
    # lemniscata (infinito) à direita
    t=np.linspace(0,2*np.pi,400); x,y=lemniscate(t,1.9); x=x*0.95+7.7; y=y*0.95+0.95
    for i in range(len(t)-1):
        f=i/len(t); col=c1 if f<0.5 else c2
        ax.plot(x[i:i+2],y[i:i+2],c=col,lw=3.2,alpha=0.35+0.6*abs(np.sin(2*np.pi*f)),solid_capstyle="round",zorder=2)
    # nós ao longo do laço
    for k,f in enumerate(np.linspace(0.06,0.94,8)):
        i=int(f*(len(t)-1)); ax.scatter(x[i],y[i],s=60,c=[c1,c2][k%2],edgecolors="white",linewidths=0.8,zorder=3)
    # pipeline de caixas no meio
    for k in range(4+n%3):
        bx=3.55+k*0.5
        ax.add_patch(FancyBboxPatch((bx,0.78),0.34,0.34,boxstyle="round,pad=0,rounding_size=0.06",fc=[c1,c2][k%2],ec="white",lw=0.6,alpha=0.85,zorder=3))
        if k: ax.annotate("",xy=(bx,0.95),xytext=(bx-0.16,0.95),arrowprops=dict(arrowstyle="-|>",color="white",lw=0.8,alpha=0.7),zorder=3)
    ax.text(0.35,0.95,f"{n:02d}",fontsize=62,fontweight="bold",color="white",alpha=0.92,va="center",ha="left")
    ax.text(0.35,0.12,"CAPÍTULO",fontsize=9,color=c2,fontweight="bold",va="bottom")
    fig.savefig(OUT+f"banner{n}.png",bbox_inches="tight",pad_inches=0,facecolor=NAVY,dpi=170); plt.close(fig); print("banner",n,flush=True)

def cover():
    W,H=6.0,8.5
    fig=plt.figure(figsize=(W,H)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis("off"); fig.patch.set_facecolor(NAVY)
    yy=np.linspace(0,1,512)[:,None]; cm=LinearSegmentedColormap.from_list("c",["#070B18","#1E1B4B","#312E81","#0B1020"]); ax.imshow(yy*np.ones((1,2)),extent=[0,W,0,H],aspect="auto",cmap=cm,zorder=0,origin="lower")
    for gx in np.arange(0.2,W,0.3):
        for gy in np.arange(0.2,H,0.3): ax.plot(gx,gy,".",ms=1.4,c="white",alpha=0.07,zorder=1)
    # halo
    for r,a in [(2.6,0.04),(2.0,0.05),(1.4,0.07)]: ax.add_patch(Circle((3.0,5.6),r,fc=CYA,alpha=a,lw=0,zorder=1))
    # infinito com gradiente de cor
    t=np.linspace(0,2*np.pi,900); x,y=lemniscate(t,2.55); x=x+3.0; y=y*1.15+5.6
    cols=LinearSegmentedColormap.from_list("lc",[CYA,IND,PNK,AMB,GRN,CYA])
    for i in range(len(t)-1):
        f=i/len(t)
        ax.plot(x[i:i+2],y[i:i+2],c=cols(f),lw=9,alpha=0.18,solid_capstyle="round",zorder=2)
        ax.plot(x[i:i+2],y[i:i+2],c=cols(f),lw=4.2,solid_capstyle="round",zorder=3)
    # estações do ciclo
    nomes=["dados","treino","teste","deploy","monitor","retreino"]
    pos=[0.02,0.17,0.33,0.52,0.70,0.86]
    for k,(f,nm) in enumerate(zip(pos,nomes)):
        i=int(f*(len(t)-1)); ax.add_patch(Circle((x[i],y[i]),0.2,fc=NAVY,ec=cols(f),lw=2.5,zorder=4)); ax.add_patch(Circle((x[i],y[i]),0.09,fc=cols(f),ec="none",zorder=5))
        off=0.36 if k%2==0 else -0.36
        ax.text(x[i],y[i]+off,nm,fontsize=7.5,color="white",ha="center",va="center",alpha=0.9,zorder=6,fontweight="bold")
    # engrenagem simples (dentes)
    cx,cy=0.95,7.6
    ang=np.linspace(0,2*np.pi,17)[:-1]
    for a in ang: ax.add_patch(Rectangle((cx+0.34*np.cos(a)-0.045,cy+0.34*np.sin(a)-0.045),0.09,0.09,angle=np.degrees(a),fc=AMB,alpha=0.85,lw=0,zorder=3))
    ax.add_patch(Circle((cx,cy),0.29,fc="none",ec=AMB,lw=3,zorder=3)); ax.add_patch(Circle((cx,cy),0.1,fc=AMB,ec="none",zorder=3))
    # barras de monitor (mini gráfico)
    for k,h in enumerate([0.25,0.5,0.35,0.7,0.55,0.9]): ax.add_patch(Rectangle((4.55+k*0.14,7.25),0.09,h*0.7,fc=[GRN,CYA][k%2],alpha=0.9,lw=0,zorder=3))
    ax.plot([4.5,5.45],[7.25,7.25],c="white",alpha=0.5,lw=1,zorder=3)
    # título
    ax.text(0.45,3.0,"MLOps",fontsize=62,fontweight="bold",color="white",va="center",zorder=6)
    ax.plot([0.5,2.5],[2.35,2.35],c=AMB,lw=5,zorder=6)
    ax.text(0.5,1.95,"Do notebook à produção",fontsize=17,color=CYA,va="center",zorder=6,fontweight="bold")
    ax.text(0.5,1.55,"com Python",fontsize=14,color="#E5E7EB",va="center",zorder=6,style="italic")
    ax.text(0.5,0.85,"Araken de Medeiros Santos",fontsize=15,fontweight="bold",color="white",va="center",zorder=6)
    ax.text(W-0.35,8.2,"2026",fontsize=10,color=GRY,ha="right",va="center",zorder=6)
    fig.savefig(OUT+"capa.png",facecolor=NAVY,dpi=220); plt.close(fig); print("capa",flush=True)

# ------------------------------------------------------------------ Capítulo 1
def f1_cozinha():
    fig,ax=canvas(8.6,4.2,(0,100),(0,50))
    rbox(ax,2,4,43,40,fc="#FFF7ED",ec=AMB,lw=2,r=3); rbox(ax,55,4,43,40,fc="#EEF2FF",ec=IND,lw=2,r=3)
    ax.text(23.5,40,"Cozinha de casa",ha="center",fontweight="bold",fontsize=11,color=AMB); ax.text(76.5,40,"Cozinha industrial",ha="center",fontweight="bold",fontsize=11,color=IND)
    # casa: 1 bolo
    ax.add_patch(Ellipse((23.5,19),14,6,fc="#92400E")); ax.add_patch(Rectangle((16.5,19),14,6,fc="#B45309")); ax.add_patch(Ellipse((23.5,25),14,6,fc="#D97706")); ax.add_patch(Circle((23.5,29),1.6,fc=RED))
    ax.text(23.5,10,"1 bolo\nreceita na cabeça\nsem testes",ha="center",fontsize=8.5,va="center")
    # fábrica: esteira com bolos
    ax.add_patch(Rectangle((58,18),37,3,fc=GRY))
    for k in range(5):
        cx=62+k*7; ax.add_patch(Rectangle((cx-2.5,21),5,4,fc="#B45309")); ax.add_patch(Ellipse((cx,25),5,2.2,fc="#D97706"))
    for k,(t,c) in enumerate([("receita\nregistrada",IND),("insumos\nconfiáveis",GRN),("testes",AMB),("fiscal de\nqualidade",RED)]):
        rbox(ax,57+k*9.6,30,8.8,7,t,fc="white",ec=c,fs=6.3,lw=1.6,r=1)
    ax.text(76.5,10,"500 bolos iguais por dia\nequipe, processos, monitoramento",ha="center",fontsize=8.5,va="center")
    arrow(ax,(46,24),(54,24),INK,lw=2.2,ms=16); ax.text(50,27.5,"MLOps",ha="center",fontsize=9,fontweight="bold",color=INK)
    save(fig,"f1_cozinha")

def f1_iceberg():
    fig,ax=canvas(8.2,5.4,(0,100),(0,66))
    ax.add_patch(Rectangle((0,0),100,34,fc="#BAE6FD",alpha=0.55,lw=0)); ax.add_patch(Rectangle((0,34),100,32,fc="#F0F9FF",lw=0))
    ax.plot([0,100],[34,34],c=CYA,lw=2)
    ax.add_patch(Polygon([[38,34],[50,56],[62,34]],fc="white",ec=GRY,lw=1.5,zorder=3))
    rbox(ax,43,38,14,7,"Código\ndo modelo",fc=IND,tc="white",bold=True,fs=8.5,r=1.2,z=6)
    ax.add_patch(Polygon([[18,34],[82,34],[70,6],[30,6]],fc="#E0E7FF",ec=IND,lw=1.5,zorder=2))
    itens=[("Coleta de\ndados",30,27.5),("Validação\nde dados",50,27.5),("Atributos",70,27.5),("Gestão de\nrecursos",36,19),("Ferramentas\nde análise",56,19),("Serviço",74,19),("Monitoramento",43,10.8),("Configuração",62,10.8)]
    for t,x,y in itens: rbox(ax,x-8.2,y-3.7,16.4,7.4,t,fc="white",ec=IND,fs=7.2,lw=1.1,r=1,z=6)
    ax.text(88,50,"visível",fontsize=9,color=GRY,style="italic"); ax.text(88,24,"abaixo da\nlinha d'água",fontsize=9,color=IND,style="italic",ha="center")
    save(fig,"f1_iceberg")

def f1_infinito():
    fig,ax=plt.subplots(figsize=(8.4,4.4)); ax.axis("off"); ax.set_aspect("equal")
    t=np.linspace(0,2*np.pi,600); x,y=lemniscate(t,3.2)
    cols=LinearSegmentedColormap.from_list("lc",[IND,CYA,GRN,AMB,IND])
    for i in range(len(t)-1): ax.plot(x[i:i+2],y[i:i+2],c=cols(i/len(t)),lw=7,solid_capstyle="round")
    ax.text(-1.9,0.0,"DEV / ML",ha="center",fontsize=14,fontweight="bold",color=IND); ax.text(1.9,0.0,"OPS",ha="center",fontsize=14,fontweight="bold",color=GRN)
    nomes=["planejar","dados","treinar","validar","implantar","monitorar","aprender","retreinar"]
    for f,nm in zip(np.linspace(0.04,0.96,8),nomes):
        i=int(f*(len(t)-1)); ax.scatter(x[i],y[i],s=330,c="white",edgecolors=cols(f),linewidths=3,zorder=3)
        off=0.55 if y[i]>=0 or True else -0.55
        ax.text(x[i],y[i]+(0.62 if (i%2==0) else -0.62),nm,ha="center",va="center",fontsize=8.5,fontweight="bold")
    ax.set_xlim(-3.8,3.8); ax.set_ylim(-1.6,1.6)
    save(fig,"f1_infinito")

def f1_eixos():
    fig,ax=canvas(8,4.9,(0,100),(-4,56))
    for (cx,cy,t,c,s) in [(25,30,"CÓDIGO",IND,"scripts, pipeline,\nAPI, testes"),(75,30,"DADOS",GRN,"lotes, rótulos,\natributos, esquema"),(50,10,"MODELO",AMB,"parâmetros,\nhiperparâmetros")]:
        ax.add_patch(Circle((cx,cy),17 if cy>20 else 14,fc=c,alpha=0.18,ec=c,lw=2.2)); ax.text(cx,cy+3,t,ha="center",fontsize=12,fontweight="bold",color=c); ax.text(cx,cy-4.5,s,ha="center",fontsize=7.8,va="center")
    ax.text(50,31,"mudou\nalgo?",ha="center",va="center",fontsize=9,fontweight="bold",color=RED)
    rbox(ax,28,47,44,7,"CACE: mudar qualquer coisa muda tudo",fc="#FEE2E2",ec=RED,fs=9.5,bold=True,r=1.5)
    for p,q in [((38,40),(45,34)),((62,40),(55,34))]: arrow(ax,p,q,RED,lw=1.4,ms=10)
    save(fig,"f1_eixos")

# ------------------------------------------------------------------ Capítulo 2
def f2_ciclo():
    fig,ax=plt.subplots(figsize=(7.2,6.2)); ax.axis("off"); ax.set_aspect("equal")
    nomes=["Requisitos","Coleta","Limpeza","Rotulagem","Atributos","Treino","Avaliação","Implantação","Monitoramento"]
    cols=[IND,IND,CYA,CYA,CYA,GRN,GRN,AMB,RED]
    n=len(nomes); R=2.6
    ang=np.pi/2-np.arange(n)*2*np.pi/n
    for i in range(n):
        a1,a2=ang[i],ang[(i+1)%n]
        arrow(ax,(R*np.cos(a1)*0.97,R*np.sin(a1)*0.97),(R*np.cos(a2)*0.97,R*np.sin(a2)*0.97),GRY,lw=1.4,rad=-0.28,ms=12)
    for i,(nm,c) in enumerate(zip(nomes,cols)):
        x,y=R*np.cos(ang[i]),R*np.sin(ang[i])
        ax.add_patch(Circle((x,y),0.52,fc=c,ec="white",lw=2.5,zorder=3)); ax.text(x,y,str(i+1),ha="center",va="center",color="white",fontweight="bold",fontsize=13,zorder=4)
        ax.text(x*1.55,y*1.32,nm,ha="center",va="center",fontsize=9.5,fontweight="bold",color=c)
    ax.text(0,0.3,"CICLO DE VIDA",ha="center",fontsize=13,fontweight="bold",color=INK); ax.text(0,-0.25,"é um laço, não uma linha",ha="center",fontsize=9.5,color=GRY,style="italic")
    for (nm,a,c) in [("Projeto",(-5.4,-3.35),IND),("Desenvolvimento",(-5.4,-3.8),GRN),("Operação",(-5.4,-4.25),AMB)]:
        ax.add_patch(Rectangle((a[0],a[1]),0.28,0.28,fc=c)); ax.text(a[0]+0.4,a[1]+0.14,nm,va="center",fontsize=8.5)
    ax.set_xlim(-5.6,5.6); ax.set_ylim(-4.6,4.0)
    save(fig,"f2_ciclo")

def f2_divida():
    fig,ax=canvas(8.2,4.4,(0,100),(0,56))
    xs=np.linspace(6,60,40); ys=5+0.0095*(xs-6)**2
    ax.plot(xs,ys,c=RED,lw=3); ax.fill_between(xs,5,ys,color=RED,alpha=0.12)
    ax.plot([6,60],[5,5],c=GRY,lw=1.2); ax.text(33,1.5,"tempo →",ha="center",fontsize=9,color=GRY); ax.text(2,52,"custo de mudar",fontsize=9,color=GRY,va="top")
    ax.scatter([8],[5.2],s=90,c=GRN,zorder=3); ax.text(9.5,10,"atalho de hoje:\n“entrega rápida”",fontsize=8.5,color=GRN,fontweight="bold")
    ax.text(66,52,"juros pagos depois",fontsize=9.5,fontweight="bold",color=RED)
    tipos=["emaranhamento (CACE)","dados de entrada instáveis","realimentação escondida","consumidores não declarados","código de cola e configuração","mudanças no mundo externo"]
    for k,t in enumerate(tipos): rbox(ax,64,44-k*7.4,34,5.6,t,fc="white",ec=RED,fs=7.6,lw=1.2,r=1)
    save(fig,"f2_divida")

def f2_arquitetura():
    fig,ax=canvas(8.6,5.8,(0,100),(0,72))
    def caixa(t,x,y,c,w=14.5,h=11): rbox(ax,x,y,w,h,t,fc="white",ec=c,fs=7.6,lw=2,r=1.5,z=3)
    top=[("Fontes de\ndados",2,GRY),("Engenharia\nde dados",19,CYA),("Armazém de\natributos",36,CYA),("Treino e\navaliação",53,GRN),("Rastreamento\nexperimentos",70,GRN),("Registro de\nmodelos",87,AMB)]
    for t,x,c in top: caixa(t,x,52,c,w=12.5)
    for i in range(5): arrow(ax,(14.5+i*17,57.5),(19+i*17,57.5),INK,lw=1.5,ms=10)
    ax.text(50,67.5,"DESENVOLVIMENTO",ha="center",fontsize=8.5,color=GRY,fontweight="bold")
    bot=[("Orquestração\nde fluxos",2,VIO),("CI/CD",24,IND),("Serviço de\nmodelos",46,AMB),("Monitoramento",68,RED)]
    for t,x,c in bot: caixa(t,x,26,c,w=17)
    for x in (19,41,63): arrow(ax,(x,31.5),(x+5,31.5),INK,lw=1.5,ms=10)
    arrow(ax,(93,52),(54,37.5),INK,lw=1.4,ms=10,rad=-0.15)
    ax.text(80,49.5,"modelo aprovado",ha="center",fontsize=7.5,color=INK,style="italic")
    arrow(ax,(76,26),(76,22),RED,lw=0,ms=1)
    ax.annotate("",xy=(59,52),xytext=(78,37.2),arrowprops=dict(arrowstyle="-|>",color=RED,lw=1.6,ls="--",connectionstyle="arc3,rad=0.25"),zorder=5)
    ax.text(90,40,"o que se aprende\nem produção volta\nao início",ha="center",fontsize=7.6,color=RED,style="italic")
    ax.text(50,21,"OPERAÇÃO",ha="center",fontsize=8.5,color=GRY,fontweight="bold")
    rbox(ax,6,5,88,8,"Governança, segurança e rastreabilidade em toda a arquitetura",fc="#F5F3FF",ec=VIO,fs=9,r=1.5,z=3)
    save(fig,"f2_arquitetura")

def f2_maturidade():
    fig,ax=canvas(8.4,4.8,(0,100),(0,60))
    niv=[("0  Manual","notebooks,\nimplantação manual",GRY),("1  Treino\nautomatizado","pipeline repetível,\nexperimentos rastreados",CYA),("2  Entrega\nautomatizada","testes, CI/CD,\nregistro de modelos",GRN),("3  Operação\ncontínua","monitoramento,\nCT, governança",AMB)]
    for k,(t,s,c) in enumerate(niv):
        x=3+k*24; h=18+k*10
        ax.add_patch(Rectangle((x,3),23,h,fc=c,alpha=0.92,ec="white",lw=2)); ax.text(x+11.5,3+h-2.5,t,ha="center",fontsize=8.4,fontweight="bold",color="white",va="top"); ax.text(x+11.5,3+6,s,ha="center",va="center",fontsize=7,color="white")
    arrow(ax,(6,54),(94,54),INK,lw=1.5,ms=14); ax.text(50,57,"cada degrau resolve a dor do anterior",ha="center",fontsize=9,style="italic")
    save(fig,"f2_maturidade")
