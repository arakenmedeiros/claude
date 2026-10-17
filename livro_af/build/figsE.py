from icons import *
from matplotlib.patches import Circle as _Circle
from matplotlib.colors import LinearSegmentedColormap
PART={1:(IND,CYA),2:(IND,CYA),3:(IND,CYA),4:(CYA,GRN),5:(CYA,GRN),6:(CYA,GRN),7:(RED,AMB),8:(RED,AMB),9:(RED,AMB),10:(AMB,PNK),11:(AMB,PNK),12:(VIO,PNK)}

def net(ax,rng,n_nodes,xr,yr,c1,c2,center=None,s=1.0):
    """desenha uma estrela federada: servidor central e nós periféricos com arestas"""
    cx,cy=center
    ang=np.linspace(0,2*np.pi,n_nodes,endpoint=False)+rng.uniform(0,1)
    nodes=[]
    for i,a in enumerate(ang):
        r=rng.uniform(0.6,1.0)
        x=cx+np.cos(a)*xr*r; y=cy+np.sin(a)*yr*r; nodes.append((x,y))
        col=c1 if i%2==0 else c2
        ax.plot([cx,x],[cy,y],c=col,lw=1.1,alpha=0.55,zorder=2)
        ax.scatter(x,y,s=80*s,c=col,zorder=4,edgecolors="white",linewidths=0.8)
        ax.scatter(x,y,s=900*s,c=col,alpha=0.10,zorder=1,linewidths=0)
        # pacote de modelo viajando
        t=rng.uniform(0.3,0.7); ax.scatter(cx+(x-cx)*t,cy+(y-cy)*t,s=14*s,c="white",zorder=5,alpha=0.9)
    for i in range(len(nodes)):
        j=(i+1)%len(nodes); ax.plot([nodes[i][0],nodes[j][0]],[nodes[i][1],nodes[j][1]],c="white",lw=0.5,alpha=0.12,zorder=1)
    ax.scatter(cx,cy,s=420*s,c="white",zorder=5,edgecolors=c2,linewidths=3)
    ax.scatter(cx,cy,s=2600*s,c=c2,alpha=0.12,zorder=1,linewidths=0)

def banner(n):
    c1,c2=PART[n]; rng=np.random.default_rng(300+n)
    fig,ax=plt.subplots(figsize=(10,1.9)); fig.patch.set_facecolor(NAVY); ax.set_facecolor(NAVY); ax.set_xlim(0,10); ax.set_ylim(0,1.9); ax.axis("off")
    g=np.linspace(0,1,256)[None,:]; cm=LinearSegmentedColormap.from_list("b",[NAVY,"#1E1B4B",NAVY]); ax.imshow(g,extent=[0,10,0,1.9],aspect="auto",cmap=cm,zorder=0)
    ASP[0]=(10/1.9)*(1.9/10)
    # duas estrelas
    for cx,cy,k in [(5.6,0.95,5+n%3),(8.6,0.95,4+n%2)]:
        net(ax,rng,k,1.25 if cx<7 else 1.0,0.75,c1,c2,center=(cx,cy),s=0.5)
    ax.text(0.35,0.95,f"{n:02d}",fontsize=62,fontweight="bold",color="white",alpha=0.92,va="center",ha="left")
    ax.text(0.35,0.12,"CAPÍTULO",fontsize=9,color=c2,fontweight="bold",va="bottom")
    fig.savefig(OUT+f"banner{n}.png",bbox_inches="tight",pad_inches=0,facecolor=NAVY,dpi=170); plt.close(fig); print("banner",n)

def cover():
    rng=np.random.default_rng(11); W,H=6.0,8.5
    fig=plt.figure(figsize=(W,H)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis("off"); fig.patch.set_facecolor(NAVY)
    ASP[0]=1.0
    yy=np.linspace(0,1,512)[:,None]; cm=LinearSegmentedColormap.from_list("c",["#0B1020","#1E1B4B","#312E81","#0B1020"]); ax.imshow(yy*np.ones((1,2)),extent=[0,W,0,H],aspect="auto",cmap=cm,zorder=0,origin="lower")
    # rede federada: servidor no centro e três anéis de clientes
    cx,cy=3.0,5.55
    for r,a in [(0.9,0.10),(1.7,0.07),(2.5,0.05)]: ax.add_patch(_Circle((cx,cy),r,fc="none",ec="white",alpha=a*2,lw=1,zorder=1))
    cores=[IND,CYA,GRN,AMB,PNK,VIO,RED]
    pts=[]
    for ring,(r,k) in enumerate([(0.9,6),(1.7,10),(2.5,15)]):
        for i in range(k):
            a=2*np.pi*i/k+ring*0.4+rng.normal(0,0.05); rr=r*rng.uniform(0.93,1.07)
            x,y=cx+np.cos(a)*rr,cy+np.sin(a)*rr; c=cores[(i+ring)%len(cores)]; pts.append((x,y,c,ring))
    for x,y,c,ring in pts:
        if ring==0: ax.plot([cx,x],[cy,y],c=c,lw=1.6,alpha=0.8,zorder=2)
        else:
            near=min([p for p in pts if p[3]==ring-1],key=lambda p:(p[0]-x)**2+(p[1]-y)**2); ax.plot([near[0],x],[near[1],y],c=c,lw=1.0,alpha=0.55,zorder=2)
    for x,y,c,ring in pts:
        ax.scatter(x,y,s=[160,110,70][ring],c=c,zorder=4,edgecolors="white",linewidths=1.0); ax.scatter(x,y,s=[1300,800,450][ring],c=c,alpha=0.10,zorder=1,linewidths=0)
    for x,y,c,ring in pts[:6]:
        for t in (0.35,0.65): ax.scatter(cx+(x-cx)*t,cy+(y-cy)*t,s=18,c="white",zorder=5,alpha=0.9)
    ax.scatter(cx,cy,s=1500,c="white",zorder=6,edgecolors=CYA,linewidths=5); ax.scatter(cx,cy,s=9000,c=CYA,alpha=0.12,zorder=1,linewidths=0)
    # ícone servidor no centro
    for i in range(3):
        ax.add_patch(FancyBboxPatch((cx-0.22,cy-0.20+i*0.14),0.44,0.11,boxstyle="round,pad=0,rounding_size=0.03",fc=NAVY,ec=NAVY,zorder=7))
        ax.scatter(cx-0.14,cy-0.145+i*0.14,s=12,c=GRN if i!=1 else AMB,zorder=8)
    # cadeado
    ax.text(0.45,2.55,"APRENDIZADO",fontsize=34,fontweight="bold",color="white",va="center",zorder=6)
    ax.text(0.45,1.85,"FEDERADO",fontsize=34,fontweight="bold",color=CYA,va="center",zorder=6)
    ax.plot([0.47,2.2],[1.45,1.45],c=AMB,lw=4,zorder=6)
    ax.text(0.47,1.12,"Fundamentos, Prática e Fronteiras",fontsize=13.5,color="#E5E7EB",va="center",zorder=6,style="italic")
    ax.text(0.47,0.45,"Araken de Medeiros Santos",fontsize=15,fontweight="bold",color="white",va="center",zorder=6)
    ax.text(W-0.35,8.15,"2026",fontsize=10,color=GRY,ha="right",va="center",zorder=6)
    fig.savefig(OUT+"capa.png",facecolor=NAVY,dpi=220); plt.close(fig); print("capa")

if __name__=="__main__":
    cover()
    for n in range(1,13): banner(n)
