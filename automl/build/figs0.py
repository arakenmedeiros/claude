from fc import *
from matplotlib.colors import LinearSegmentedColormap
PART={1:(IND,CYA),2:(IND,CYA),3:(IND,CYA),4:(CYA,GRN),5:(CYA,GRN),6:(CYA,GRN),7:(CYA,GRN),8:(GRN,AMB),9:(GRN,AMB),10:(GRN,AMB),11:(AMB,RED),12:(AMB,RED),13:(VIO,PNK),14:(VIO,PNK)}
def landscape(rng,nx=300,ny=120,K=5):
    x=np.linspace(0,10,nx); y=np.linspace(0,1.9,ny); X,Y=np.meshgrid(x,y); Z=np.zeros_like(X)
    for _ in range(K):
        cx,cy=rng.uniform(3.8,9.5),rng.uniform(0.25,1.65); s=rng.uniform(0.5,1.4); a=rng.uniform(0.5,1)
        Z+=a*np.exp(-(((X-cx)/s)**2+((Y-cy)/(s*0.35))**2))
    return X,Y,Z
def banner(n):
    c1,c2=PART[n]; rng=np.random.default_rng(300+n)
    fig,ax=plt.subplots(figsize=(10,1.9)); fig.patch.set_facecolor(NAVY); ax.set_facecolor(NAVY); ax.set_xlim(0,10); ax.set_ylim(0,1.9); ax.axis("off")
    X,Y,Z=landscape(rng)
    cm=LinearSegmentedColormap.from_list("b",[NAVY,"#1E1B4B",c1,c2],N=256)
    ax.contourf(X,Y,Z,levels=14,cmap=cm,alpha=0.85,zorder=1); ax.contour(X,Y,Z,levels=14,colors="white",linewidths=0.25,alpha=0.25,zorder=2)
    # search path converging to maximum
    Zm=np.where(X>3.8,Z,-1); iy,ix=np.unravel_index(Zm.argmax(),Zm.shape); tx,ty=X[iy,ix],Y[iy,ix]
    P=[(4.0 if tx>6.5 else 9.6,rng.uniform(0.15,1.75))]
    for k in range(14):
        p=np.array(P[-1]); q=p+(np.array([tx,ty])-p)*rng.uniform(0.22,0.4)+rng.normal(0,[0.15,0.07])
        P.append((float(np.clip(q[0],3.8,9.9)),float(np.clip(q[1],0.05,1.85))))
    P=np.array(P); ax.plot(P[:,0],P[:,1],c="white",lw=0.9,alpha=0.55,zorder=3)
    ax.scatter(P[:,0],P[:,1],s=np.linspace(14,40,len(P)),c=[c2]*len(P),edgecolors="white",linewidths=0.6,zorder=4)
    ax.scatter([tx],[ty],s=220,marker="*",c=AMB,edgecolors="white",linewidths=0.8,zorder=5)
    ax.text(0.35,0.95,f"{n:02d}",fontsize=62,fontweight="bold",color="white",alpha=0.95,va="center",ha="left",zorder=6)
    ax.text(0.35,0.12,"CAPÍTULO",fontsize=9,color=c2,fontweight="bold",va="bottom",zorder=6)
    fig.savefig(OUT+f"banner{n}.png",bbox_inches="tight",pad_inches=0,facecolor=NAVY,dpi=170); plt.close(fig); print("banner",n,flush=True)

def cover():
    rng=np.random.default_rng(11); W,H=6.0,8.5
    fig=plt.figure(figsize=(W,H)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis("off"); fig.patch.set_facecolor(NAVY)
    yy=np.linspace(0,1,512)[:,None]; cm=LinearSegmentedColormap.from_list("c",["#070B18","#1E1B4B","#312E81","#0B1020"]); ax.imshow(yy*np.ones((1,2)),extent=[0,W,0,H],aspect="auto",cmap=cm,zorder=0,origin="lower")
    # background search landscape (top half)
    x=np.linspace(0,W,260); y=np.linspace(3.2,H,260); X,Y=np.meshgrid(x,y); Z=np.zeros_like(X)
    peaks=[(4.4,6.5,0.9,1.0),(1.5,5.2,0.8,0.7),(3.0,4.2,0.7,0.5),(5.2,4.6,0.6,0.45),(2.2,7.4,0.7,0.5)]
    for cx,cy,s,a in peaks: Z+=a*np.exp(-(((X-cx)/s)**2+((Y-cy)/s)**2))
    cm2=LinearSegmentedColormap.from_list("l",["#0B1020","#1E1B4B",IND,CYA,"#A5F3FC"],N=256)
    ax.contourf(X,Y,Z,levels=18,cmap=cm2,alpha=0.55,zorder=1); ax.contour(X,Y,Z,levels=18,colors="white",linewidths=0.35,alpha=0.35,zorder=2)
    # optimisation trajectory converging to the main peak
    t=np.array([4.4,6.5]); p=np.array([0.7,3.6]); path=[p.copy()]
    for k in range(13):
        p=p+(t-p)*rng.uniform(0.2,0.38)+rng.normal(0,[0.18,0.18]); path.append(p.copy())
    path=np.array(path); ax.plot(path[:,0],path[:,1],c="white",lw=1.3,alpha=0.8,zorder=4)
    ax.scatter(path[:,0],path[:,1],s=np.linspace(25,110,len(path)),c=CYA,edgecolors="white",linewidths=0.8,zorder=5)
    ax.scatter([4.4],[6.5],s=620,marker="*",c=AMB,edgecolors="white",linewidths=1.2,zorder=6)
    for r,a in [(0.5,0.12),(0.85,0.07),(1.3,0.04)]: ax.add_patch(Circle((4.4,6.5),r,fc=AMB,alpha=a,lw=0,zorder=3))
    # pipeline blocks along the lower band
    labels=["DADOS","ATRIBUTOS","BUSCA","MODELO"]; cols=[IND,CYA,GRN,AMB]
    for i,(lb,cl) in enumerate(zip(labels,cols)):
        x0=0.35+i*1.38
        rbox(ax,x0,3.0,1.15,0.5,lb,fc=cl,ec="white",fs=7.5,tc="white",bold=True,r=0.1,lw=0.8,alpha=0.95)
        if i<3: arrow(ax,(x0+1.17,3.25),(x0+1.36,3.25),c="white",lw=1.2,ms=8)
    ax.text(0.45,2.15,"AutoML",fontsize=54,fontweight="bold",color="white",va="center",zorder=7)
    ax.plot([0.47,2.3],[1.62,1.62],c=AMB,lw=4,zorder=7)
    ax.text(0.47,1.25,"Teoria e Prática",fontsize=19,color=CYA,va="center",zorder=7,style="italic",fontweight="bold")
    ax.text(0.47,0.62,"Araken de Medeiros Santos",fontsize=15,fontweight="bold",color="white",va="center",zorder=7)
    ax.text(W-0.35,8.18,"2026",fontsize=10,color=GRY,ha="right",va="center",zorder=7)
    fig.savefig(OUT+"capa.png",facecolor=NAVY,dpi=220); plt.close(fig); print("capa",flush=True)
if __name__=="__main__":
    cover()
    for n in range(1,15): banner(n)
