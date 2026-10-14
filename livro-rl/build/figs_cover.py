from fc import *
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import RegularPolygon

def star(ax,x,y,r,c=AMB,z=6,alpha=1):
    pts_=[]
    for k in range(10):
        ang=np.pi/2+k*np.pi/5; rr=r if k%2==0 else r*0.45
        pts_.append((x+rr*np.cos(ang),y+rr*np.sin(ang)))
    ax.add_patch(Polygon(pts_,closed=True,fc=c,ec="white",lw=0.6,zorder=z,alpha=alpha))

def value_grid(ax,x0,y0,nx,ny,cell,goal,traps,rng,glow=True,path=None,alpha=0.9):
    """desenha uma grade colorida pelo 'valor' (maior perto do objetivo)."""
    cm=LinearSegmentedColormap.from_list("v",["#1E1B4B","#312E81","#4F46E5","#06B6D4","#10B981","#FDE68A"])
    for i in range(nx):
        for j in range(ny):
            d=abs(i-goal[0])+abs(j-goal[1]); v=np.exp(-d/4.2)+rng.normal(0,0.03)
            for t in traps:
                if (i,j)==t: v=-1
            col=cm(np.clip(v,0,1)) if v>=0 else "#7F1D1D"
            ax.add_patch(Rectangle((x0+i*cell,y0+j*cell),cell*0.94,cell*0.94,fc=col,ec="none",alpha=alpha*(0.55+0.45*max(v,0)),zorder=2))
    if path is not None:
        P=np.array([(x0+(i+0.47)*cell,y0+(j+0.47)*cell) for i,j in path])
        for lw,a in [(9,0.10),(5,0.2),(2.2,0.95)]: ax.plot(P[:,0],P[:,1],c="white" if lw<3 else CYA,lw=lw,alpha=a,zorder=4,solid_capstyle="round")
        ax.scatter(P[:,0],P[:,1],s=14,c="white",zorder=5)

def make_path(start,goal,traps,rng):
    p=[start]; cur=start
    while cur!=goal:
        dx=np.sign(goal[0]-cur[0]); dy=np.sign(goal[1]-cur[1])
        opts=[]
        if dx: opts.append((cur[0]+dx,cur[1]))
        if dy: opts.append((cur[0],cur[1]+dy))
        cur=opts[rng.integers(len(opts))]; p.append(cur)
    return p

def cover():
    rng=np.random.default_rng(11); W,H=6.0,8.5
    fig=plt.figure(figsize=(W,H)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis("off"); fig.patch.set_facecolor(NAVY)
    yy=np.linspace(0,1,512)[:,None]; cm=LinearSegmentedColormap.from_list("c",["#0B1020","#1E1B4B","#312E81","#0B1020"])
    ax.imshow(yy*np.ones((1,2)),extent=[0,W,0,H],aspect="auto",cmap=cm,zorder=0,origin="lower")
    # estrelas de fundo
    S=rng.uniform(0,1,(160,2))*[W,H]; ax.scatter(S[:,0],S[:,1],s=rng.uniform(1,8,160),c="white",alpha=0.25,lw=0,zorder=1)
    nx,ny,cell=9,8,0.60; x0,y0=0.45,3.35
    goal=(7,6); traps=[(3,2),(4,2),(5,2),(2,5),(6,3),(5,5)]
    for (a,b) in traps: pass
    value_grid(ax,x0,y0,nx,ny,cell,goal,traps,rng)
    path=[(0,0),(0,1),(1,1),(1,2),(1,3),(2,3),(3,3),(3,4),(4,4),(4,5),(4,6),(5,6),(6,6),(7,6)]
    P=np.array([(x0+(i+0.47)*cell,y0+(j+0.47)*cell) for i,j in path])
    for lw,a in [(11,0.10),(6,0.22),(2.4,0.95)]: ax.plot(P[:,0],P[:,1],c=CYA if lw>3 else "white",lw=lw,alpha=a,zorder=4,solid_capstyle="round")
    ax.scatter(P[:-1,0],P[:-1,1],s=16,c="white",zorder=5)
    # agente
    ax.add_patch(Circle(P[0],0.17,fc=PNK,ec="white",lw=1.2,zorder=7)); ax.add_patch(Circle(P[0],0.34,fc=PNK,alpha=0.25,zorder=6))
    # objetivo
    gx,gy=x0+(goal[0]+0.47)*cell,y0+(goal[1]+0.47)*cell
    for r,a in [(0.7,0.08),(0.5,0.12),(0.32,0.2)]: ax.add_patch(Circle((gx,gy),r,fc=AMB,alpha=a,zorder=3,lw=0))
    star(ax,gx,gy,0.28)
    # armadilhas
    for (i,j) in traps:
        cx,cy=x0+(i+0.47)*cell,y0+(j+0.47)*cell
        ax.plot([cx-0.12,cx+0.12],[cy-0.12,cy+0.12],c="#FCA5A5",lw=2,zorder=6); ax.plot([cx-0.12,cx+0.12],[cy+0.12,cy-0.12],c="#FCA5A5",lw=2,zorder=6)
    # recompensas flutuantes
    for (tx,ty,s,c) in [(gx+0.1,gy+0.75,"+1",AMB),(x0+(3+0.47)*cell,y0+(2+0.47)*cell+0.5,"−1","#FCA5A5"),(P[6][0]+0.05,P[6][1]+0.36,"Q(s,a)",CYA)]:
        ax.text(tx,ty,s,color=c,fontsize=11,fontweight="bold",ha="center",va="center",zorder=8)
    # título
    ax.text(0.45,2.55,"APRENDIZADO",fontsize=34,fontweight="bold",color="white",va="center",zorder=6)
    ax.text(0.45,1.88,"POR REFORÇO",fontsize=34,fontweight="bold",color=AMB,va="center",zorder=6)
    ax.plot([0.47,2.2],[1.5,1.5],c=CYA,lw=4,zorder=6)
    ax.text(0.47,1.12,"Como agentes aprendem a decidir: da intuição\nao aprendizado profundo, com Python",fontsize=11.5,color="#E5E7EB",va="center",zorder=6,style="italic",linespacing=1.4)
    ax.text(0.47,0.42,"Araken de Medeiros Santos",fontsize=15,fontweight="bold",color="white",va="center",zorder=6)
    ax.text(W-0.35,8.2,"2026",fontsize=10,color=GRY,ha="right",va="center",zorder=6)
    fig.savefig(OUT+"capa.png",facecolor=NAVY,dpi=220); plt.close(fig); print("capa")

PART={1:(IND,CYA),2:(IND,CYA),3:(IND,CYA),4:(IND,CYA),5:(CYA,GRN),6:(CYA,GRN),7:(GRN,AMB),8:(GRN,AMB),9:(GRN,AMB),10:(GRN,AMB),11:(AMB,RED),12:(AMB,RED),13:(AMB,RED),14:(RED,PNK)}
def banner(n):
    c1,c2=PART[n]; rng=np.random.default_rng(300+n)
    fig,ax=plt.subplots(figsize=(10,1.9)); fig.patch.set_facecolor(NAVY); ax.set_facecolor(NAVY); ax.set_xlim(0,10); ax.set_ylim(0,1.9); ax.axis("off")
    g=np.linspace(0,1,256)[None,:]; cm=LinearSegmentedColormap.from_list("b",[NAVY,"#1E1B4B",NAVY]); ax.imshow(g,extent=[0,10,0,1.9],aspect="auto",cmap=cm,zorder=0)
    nx,ny,cell=11,5,0.33; x0,y0=4.6,0.12
    goal=(int(rng.integers(8,11)),int(rng.integers(1,5))); traps=[(int(rng.integers(2,9)),int(rng.integers(0,5))) for _ in range(3)]
    traps=[t for t in traps if t!=goal]
    vc=LinearSegmentedColormap.from_list("v",["#1E1B4B",c1,c2,"#FDE68A"])
    for i in range(nx):
        for j in range(ny):
            d=abs(i-goal[0])+abs(j-goal[1]); v=np.exp(-d/3.5)
            col="#7F1D1D" if (i,j) in traps else vc(np.clip(v,0,1))
            ax.add_patch(Rectangle((x0+i*cell,y0+j*cell),cell*0.92,cell*0.92,fc=col,ec="none",alpha=0.35+0.55*v if (i,j) not in traps else 0.8))
    path=make_path((0,int(rng.integers(0,5))),goal,traps,rng)
    P=np.array([(x0+(i+0.46)*cell,y0+(j+0.46)*cell) for i,j in path])
    ax.plot(P[:,0],P[:,1],c="white",lw=1.6,alpha=0.9,zorder=4); ax.plot(P[:,0],P[:,1],c=c2,lw=6,alpha=0.18,zorder=3)
    ax.add_patch(Circle(P[0],0.09,fc=PNK,ec="white",lw=0.8,zorder=6)); star(ax,P[-1][0],P[-1][1],0.14)
    for (i,j) in traps:
        cx,cy=x0+(i+0.46)*cell,y0+(j+0.46)*cell; ax.plot([cx-.06,cx+.06],[cy-.06,cy+.06],c="#FCA5A5",lw=1.2,zorder=6); ax.plot([cx-.06,cx+.06],[cy+.06,cy-.06],c="#FCA5A5",lw=1.2,zorder=6)
    lab="PROJETO" if n==14 else "CAPÍTULO"
    ax.text(0.35,0.95,f"{n:02d}" if n<14 else "★",fontsize=62,fontweight="bold",color="white",alpha=0.92,va="center",ha="left")
    ax.text(0.35,0.12,lab,fontsize=9,color=c2,fontweight="bold",va="bottom")
    fig.savefig(OUT+f"banner{n}.png",bbox_inches="tight",pad_inches=0,facecolor=NAVY,dpi=170); plt.close(fig); print("banner",n)

if __name__=="__main__":
    cover()
    for n in range(1,15): banner(n)
